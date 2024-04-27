# 用于连续动作的PPO
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal


# ------------------------------------- #
# 策略网络--输出连续动作的高斯分布的均值和标准差
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc_mu = nn.Linear(n_hiddens, n_actions)
        self.fc_std = nn.Linear(n_hiddens, n_actions)
        # Initialize layers
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc_mu.weight)
        torch.nn.init.xavier_uniform_(self.fc_std.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2 * torch.tanh(self.fc_mu(x))  # Keeping output range [-2, 2]
        std = F.softplus(self.fc_std(x)) + 1e-8  # Adding epsilon to avoid std = 0
        return mu, std


# ------------------------------------- #
# 价值网络 -- 评估当前状态的价值
# ------------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]
        return x


# ------------------------------------- #
# 模型构建--处理连续动作
# ------------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 属性分配
        self.lmbda = lmbda  # GAE优势函数的缩放因子
        self.epochs = epochs  # 一条序列的数据用来训练多少轮
        self.eps = eps  # 截断范围
        self.gamma = gamma  # 折扣系数
        self.device = device

        # 动作选择

    def take_action(self, state):  # 输入当前时刻的状态
        # Ensure the state is a numpy array and convert it to a PyTorch tensor with the correct shape
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)  # Convert state to float
        # 预测当前状态的动作，输出动作概率的高斯分布
        mu, std = self.actor(state)
        # 构造高斯分布
        dist = torch.distributions.Normal(mu, std)
        # 随机选择动作
        action = dist.sample().squeeze(0).cpu().numpy()  # Ensure action is properly shaped and back to numpy
        return action  # 返回动作值

    # 训练
    def update(self, transition_dict):
        # Load data into tensors and transfer to the device
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        # Compute the value function targets and temporal differences
        with torch.no_grad():
            next_state_values = self.critic(next_states).detach()
            targets = rewards + self.gamma * next_state_values * (1 - dones)

        values = self.critic(states)
        td_error = targets - values
        value_loss = F.mse_loss(values, targets)

        # Backpropagate on the critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Gradient clipping
        self.critic_optimizer.step()

        # Update the policy network
        mu, std = self.actor(states)
        std = torch.clamp(std, min=1e-8)  # Preventing zero standard deviation
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
        old_log_probs = log_probs.detach()

        advantages = (td_error.detach()).squeeze()

        for _ in range(self.epochs):
            # Recompute distributions for new policy
            mu, std = self.actor(states)
            std = torch.clamp(std, min=1e-8)  # Ensuring std is not zero
            dist = Normal(mu, std)
            new_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)

            # Calculate the ratio (pi_theta / pi_theta_old)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            # Backpropagate on the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Gradient clipping
            self.actor_optimizer.step()

        # Optional: Check for NaNs and handle them
        if torch.isnan(actor_loss) or torch.isnan(value_loss):
            print("Warning: NaN detected in loss calculations")
            # Add more handling logic here if needed (e.g., reducing learning rate, reverting to a previous checkpoint)