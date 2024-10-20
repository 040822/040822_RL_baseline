import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import gym  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # 由于模型简单，cpu训练速度反而更快
print("device is:",device)

# 定义策略网络
class PolicyNetwork(nn.Module):
    """
    策略网络。输入状态，输出动作的概率分布。
    Args:
        input_size (int): The size of the input layer.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output layer.
    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
    Methods:
        forward(x): Performs a forward pass through the network.
        act(state): Selects an action based on the given state.

    """

    def __init__(self, input_size, hidden_size, output_size):
        # 一个简单的MLP
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1) # softmax函数特性：输出值在0-1之间，且和为1 =》概率分布
        )
    
    def forward(self, x):
        # 前向传播
        x = self.fc(x)
        return x
    

# 定义策略梯度算法
class REINFORCE:
    def __init__(self, config):
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        output_size = config['output_size']
        lr = config['learning_rate']
        self.gamma = config['gamma']
        
        self.policy_net = PolicyNetwork(input_size, hidden_size, output_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.rewards = []
        self.actions = []
        self.log_probs = []
    
    def reset(self):
        self.rewards = []
        self.actions = []
        self.log_probs = []

    def update(self):
        # 计算回报的累积奖励,也就是return
        cumulative_reward = 0
        cumulative_rewards = [] #return
        loss = []
        
        #for reward in reversed(self.rewards): 
        for reward in self.rewards[::-1]: # 从后往前遍历
            cumulative_reward = reward + cumulative_reward*self.gamma
            cumulative_rewards.insert(0, cumulative_reward) # 从头部插入。注意rewards列表是翻转之后操作的。
        
        # 确保 cumulative_rewards 和 self.log_probs 启用了梯度计算
        #cumulative_rewards_tensor = torch.tensor(cumulative_rewards, requires_grad=True)
        cumulative_rewards_tensor = torch.tensor(cumulative_rewards)
        #log_probs_tensor = torch.tensor(self.log_probs)
        
        cumulative_rewards_tensor = (cumulative_rewards_tensor - cumulative_rewards_tensor.mean()) / (cumulative_rewards_tensor.std() + 1e-5) # 归一化
        
        for log_prob, R in zip(self.log_probs, cumulative_rewards_tensor):
            loss.append(-log_prob * R)
        
        
        
        #loss = - cumulative_rewards_tensor * log_probs_tensor
        loss = torch.stack(loss).sum()
        # L = - return * log(π)
        
        # 更新策略网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    
    def select_action(self, state):
        # 选择动作，输入状态，输出动作
        
        state = self.state_process(state) # 对输入的state进行处理
        probs = self.policy_net(state) # 前向传播，让神经网络输出概率分布
        
        # 从概率分布中采样一个动作
        prob_distribution = Categorical(probs)
        action = prob_distribution.sample()
        log_prob = prob_distribution.log_prob(action)
        
        return action.item(), log_prob
    
    def state_process(self, state):
        # 对输入的state进行处理，比如将numpy数组转换为tensor
        # 这个函数因环境而异，或许不应该写在算法类里而应该和train函数一样写在外面？
        # return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        return torch.from_numpy(state).float()
    
    
def train(env, agent, num_episodes):
    """
    实现env和agent的交互，并调用agent的update函数进行学习。

    Args:
        env: The environment to interact with.
        agent: The agent to train.
        num_episodes: The number of episodes to train the agent.

    Returns:
        None
    """
    writer = SummaryWriter()
    start_time = time.time()  # 记录开始时间
    print_every = num_episodes/100 # 每训练1%输出一次信息
    print_every = 1
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset()
        while True:
            action,log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.rewards.append(reward)
            agent.actions.append(action)
            agent.log_probs.append(log_prob)
            
            if done: # 如果游戏结束，跳出循环
                break
            state = next_state
        
        loss = agent.update()
        reward_sum = sum(agent.rewards)
        now_time = time.time()
        time_elapsed = now_time - start_time
        time_last = time_elapsed / (episode + 1) * (num_episodes - episode - 1) # 预计剩余时间
        if episode % print_every == 0:
            print('Episode: {}, Loss: {}, Reward: {}'.format(episode, loss,reward_sum))
            print('Time elapsed: {:.2f} s, Time remaining: {:.2f} s'.format(time_elapsed, time_last))
  
        writer.add_scalar('Loss', loss.item(), episode)
        writer.add_scalar('Reward', reward_sum, episode)
    
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f'Training completed in {elapsed_time:.2f} seconds')
    return None
    
# 使用示例
if __name__ == '__main__':
    with open('./REINFORCE/config.json', 'r') as f: # 读取配置文件
        config = json.load(f)
    env = gym.make(config['env_name'])
    num_episodes = int(config['num_episodes'])
    
    # 获取环境参数,根据环境参数定义输入输出维度
    config['input_size'] = env.observation_space.shape[0] # 4
    config['output_size'] = env.action_space.n # 2
    
    # 实例化agent
    agent = REINFORCE(config)
    # 运行train函数
    train(env, agent, num_episodes)
