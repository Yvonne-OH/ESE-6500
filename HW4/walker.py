# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:59:18 2024

@author: 39829
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dm_control import suite
import matplotlib.pyplot as plt
from dm_control import viewer

# 设置允许多个OpenMP运行时
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(128, 128), activation=nn.Tanh):
        super().__init__()
        self.mu_net = mlp([input_dim] + list(hidden_sizes) + [output_dim], activation)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        mu = self.mu_net(x)
        std = torch.exp(self.log_std)
        return mu, std

class MLPValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.v_net = mlp([input_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, x):
        return self.v_net(x).squeeze(-1)

def rollout(env, actor, steps=1000):
    trajectory = []
    time_step = env.reset()
    x=time_step.observation
    
    #print (x['orientations'].tolist())
    
    obs = np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
    
    #print(time_step.observation)

    for _ in range(steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mu, std = actor(obs_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        time_step = env.step(action.numpy())
        reward = time_step.reward
        #next_obs = np.concatenate([np.atleast_1d(time_step.observation[k]) for k in time_step.observation])
        x=time_step.observation
        next_obs = np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
        
        done = time_step.last()

        trajectory.append((obs, action.numpy(), reward, next_obs, done, log_prob.numpy()))
        if done:
            break
        obs = next_obs
    return trajectory

def compute_returns(rewards, mask, gamma=0.99):
    R = 0
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * mask[step]
        returns.insert(0, R)
    return returns

def ppo_update(actor, critic, data, optimizer, clip_param=0.2):
    obs, actions, log_probs_old, returns, advantages = data
    obs = torch.as_tensor(obs, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    log_probs_old = torch.as_tensor(log_probs_old, dtype=torch.float32)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-7)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    
    mu, std = actor(obs)
    dist = torch.distributions.Normal(mu, std)
    log_probs = dist.log_prob(actions).sum(dim=-1)
    ratios = (log_probs - log_probs_old).exp()

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
    #policy_loss = -torch.min(surr1, surr2).mean()
    policy_loss =  -torch.min(surr1, surr2).mean()
    
    
    values = critic(obs)
    value_loss = F.mse_loss(values, returns)
    #print(returns)

    optimizer.zero_grad()
    (policy_loss + 0.00001*value_loss ).backward()
    optimizer.step()

# Environment setup
r0 = np.random.RandomState(42)
env = suite.load(domain_name="walker", task_name="stand",task_kwargs={'random': r0})
observation_spec = env.observation_spec()
observation_dim = sum(np.prod(obs_spec.shape) for obs_spec in observation_spec.values())
action_spec = env.action_spec()
action_dim = np.prod(action_spec.shape)
# Convert to int explicitly
observation_dim = int(sum(np.prod(obs_spec.shape) for obs_spec in observation_spec.values()))
action_dim = int(np.prod(action_spec.shape))


# Model and optimizer
actor = MLPActor(observation_dim, action_dim)
critic = MLPValueFunction(observation_dim)
optimizer = optim.Adam((list(actor.parameters()) + list(critic.parameters())), lr=1e-3)

# Training loop
rewards = []  # To store rewards for plotting

for i_episode in range(50000):  # Adjust this to train for more episodes
    trajectory = rollout(env, actor)
    observations, actions, rewards_episode, next_observations, dones, log_probs = zip(*trajectory)
    mask = [0 if done else 1 for done in dones]
    returns = compute_returns(rewards_episode, mask)
    
    advantages = [ret - val for ret, val in zip(returns, critic(torch.tensor(observations, dtype=torch.float32)))]
    data = (observations, actions, log_probs, returns, advantages)
    
    ppo_update(actor, critic, data, optimizer)
    
    total_reward = sum(rewards_episode)
    rewards.append(total_reward)
    
    if i_episode % 100 == 0:
        print(f"Episode {i_episode}, Total Reward: {total_reward}")
        total_reward =0
        plt.plot(rewards, label='Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards Over Time')
        plt.legend()
        plt.show()

# Save trained models
torch.save(actor.state_dict(), "ppo_actor.pth")
torch.save(critic.state_dict(), "ppo_critic.pth")

# Load and visualize with viewer
actor.load_state_dict(torch.load('ppo_actor.pth'))
actor.eval()

def policy(time_step):
    obs = np.concatenate([np.atleast_1d(time_step.observation[k]) for k in time_step.observation])
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = actor(obs_tensor)
    return mu.numpy()

viewer.launch(env, policy)