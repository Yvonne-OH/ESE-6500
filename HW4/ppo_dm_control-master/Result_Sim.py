# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:25:41 2024

@author: 39829
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from torch.optim import Adam
import torch.nn as nn
import torch

from copy import copy
import pandas as pd
import numpy as np
import os
from dm_control import suite
import matplotlib.pyplot as plt
from dm_control import viewer
from dm_env import specs
from ppo import PPO


# Load data
data = pd.read_csv('./training/walker/walk/training_log (1).csv')

loss = data['Total Reward']

# Define a function to remove outliers using a rolling window approach
def remove_outliers_rolling(data, window_size=20):
    rolling_median = data.rolling(window=window_size, center=True).median()
    rolling_std = data.rolling(window=window_size, center=True).std()
    lower_bound = rolling_median - 0.5 * rolling_std
    upper_bound = rolling_median + 0.5 * rolling_std
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Remove outliers with a rolling window
filtered_loss_rolling = remove_outliers_rolling(loss).dropna()

# Apply Savitzky-Golay filter for smoothing
smoothed_loss_rolling = savgol_filter(filtered_loss_rolling, 81, 2)  # window size 51, polynomial order 3

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(loss.index, loss, 'b', alpha=0.1, label='Original Total Reward')
plt.plot(filtered_loss_rolling.index, filtered_loss_rolling, 'g', alpha=0.45, label='Filtered Total Reward')
plt.plot(filtered_loss_rolling.index, smoothed_loss_rolling, 'r', label='Smoothed Total Reward')
plt.title('Training Reward Analysis')
plt.xlabel('Epoch')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Initialize a random environment with a set seed
r0 = np.random.RandomState(42)
env = suite.load(domain_name="walker", task_name="walk", task_kwargs={'random': r0})

# Helper function to calculate the total dimension of observation spaces
def calculate_total_dim(specs):
    return sum(np.prod(spec.shape) for spec in specs.values())

# Calculate observation and action dimensions
observation_spec = env.observation_spec()
observation_dim = int(calculate_total_dim(observation_spec))
action_spec = env.action_spec()
action_dim = np.prod(action_spec.shape)

# Define an environment wrapper to standardize the interface
class EnvironmentWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def reset(self):
        time_step = self.env.reset()
        return self._process_observations(time_step.observation)

    def step(self, action):
        time_step = self.env.step(action)
        observation = self._process_observations(time_step.observation)
        reward = time_step.reward
        done = time_step.last()
        return observation, reward, done, {}

    def _process_observations(self, observations):
        # Flatten and concatenate all observations into a single tensor
        processed = np.concatenate([np.ravel(observations[key]) for key in observations])
        return processed

wrapped_env = EnvironmentWrapper(env)

# Initialize the PPO with the wrapped environment
hyperparameters = {
    'create_new_training_data': True,
    'load_model': False,
    'training_path': './training/walker/walk'
}

ppo_agent = PPO(wrapped_env, **hyperparameters)

# Load a trained model, assuming it's already trained
ppo_agent.ac_model.load_state_dict(torch.load('./training/walker/walk/ppo_ac_model.pth'))
ppo_agent.ac_model.eval()

# Define the policy function for DM Control Viewer
def policy(time_step):
    if time_step.first():
        action = np.zeros(env.action_spec().shape)
    else:
        observation = wrapped_env._process_observations(time_step.observation)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action_tensor, _, _ = ppo_agent.ac_model.step(obs_tensor)
        # Check if action_tensor is already a NumPy array or a PyTorch tensor
        if isinstance(action_tensor, np.ndarray):
            action = action_tensor
        else:
            action = action_tensor.numpy()  # Convert only if it's a tensor
    return action
# Launch the viewer
viewer.launch(env, policy)
