import torch

import torch.nn as nn
import torch.optim as optim
import gym  

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
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
    # 前向传播
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x
    
    def act(self, state):
    # 选择动作，输入状态，输出动作
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state).cpu()
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action)

# 定义策略梯度算法
class REINFORCE:
    def __init__(self, lr):
        self.policy_net = policy_net(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.rewards = []
        self.actions = []
    
    def reset(self):
        self.rewards = []
        self.actions = []
        self.log_probs = []

    def update(self,env, num_episodes, learning_rate):
        for episode in range(num_episodes):


            # 计算回报的累积奖励
            cumulative_rewards = []
            cumulative_reward = 0
            for reward in reversed(episode_rewards):
                cumulative_reward = reward + cumulative_reward
                cumulative_rewards.insert(0, cumulative_reward)

            # 更新策略网络
            self.optimizer.zero_grad()
            loss = 0
            for reward, action in zip(cumulative_rewards, episode_actions):
                loss -= reward * torch.log(action_probs[:, action])
            loss.backward()
            self.optimizer.step()

        return policy_net
    def select_action(self, state):
        return self.policy_net.act(state)
    
    
def train(env, policy_net, num_episodes, learning_rate):
    state = env.reset()
    agent = REINFORCE()
    while True:
        #state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.rewards.append(reward)
        agent.actions.append(action)
        if done:
            break
        state = next_state
    
# 使用示例
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 128
    learning_rate = 0.001
    num_episodes = 1000

    policy_net = PolicyNetwork(input_size, hidden_size, output_size)
    trained_policy_net = train(env, policy_net, num_episodes, learning_rate)
