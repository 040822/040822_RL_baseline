import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# 策略梯度下降训练
def train():
    env = gym.make('CartPole-v1')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    policy_net = PolicyNetwork(state_space, action_space)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    writer = SummaryWriter()

    def select_action(state):
        state = torch.from_numpy(state).float()
        probs = policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def finish_episode(episode_rewards, log_probs):
        R = 0
        policy_loss = []
        returns = []
        
        # 反向计算每个时间步的回报
        for r in episode_rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        #policy_loss = torch.cat(policy_loss).sum()
        #policy_loss = [loss.unsqueeze(0) for loss in policy_loss]
        policy_loss = torch.stack(policy_loss).sum()

        policy_loss.backward()
        print("policy_loss: ", policy_loss.item())
        optimizer.step()

    for episode in range(100):
        state = env.reset()
        episode_rewards = []
        log_probs = []
        for t in range(1, 10000):  # 假设最大步数为10000
            action, log_prob = select_action(state)
            state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            episode_rewards.append(reward)
            if done:
                break
        
        finish_episode(episode_rewards, log_probs)
        total_reward = sum(episode_rewards)
        writer.add_scalar('Reward', total_reward, episode)
        print(f'Episode {episode}\tReward: {total_reward}')
    
    writer.close()
    env.close()

if __name__ == '__main__':
    train()