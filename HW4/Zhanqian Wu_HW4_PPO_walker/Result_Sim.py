# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:25:41 2024
@author: Zhanqian Wu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import torch
from dm_control import suite
from dm_control import viewer
from ppo import PPO

# Load data from CSV file
data = pd.read_csv('./training/walker/walk/best_training.csv')

# Extract the 'Total Reward' column
loss = data['Total Reward']

# Define a function to remove outliers using a rolling window approach
def remove_outliers_rolling(data, window_size=20):
    """
    Removes outliers from the data using a rolling window median and standard deviation approach.
    Args:
        data: Pandas Series containing the data.
        window_size: Size of the rolling window.
    Returns:
        Filtered data with outliers removed.
    """
    rolling_median = data.rolling(window=window_size, center=True).median()
    rolling_std = data.rolling(window=window_size, center=True).std()
    lower_bound = rolling_median - 0.5 * rolling_std
    upper_bound = rolling_median + 0.5 * rolling_std
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Remove outliers and drop missing values
filtered_loss_rolling = remove_outliers_rolling(loss).dropna()

# Apply Savitzky-Golay filter to smooth the filtered data
smoothed_loss_rolling = savgol_filter(filtered_loss_rolling, 81, 2)  # window size 81, polynomial order 2

# Plot the original, filtered, and smoothed total rewards
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

# Initialize a random environment with a set seed for reproducibility
r0 = np.random.RandomState(42)
env = suite.load(domain_name="walker", task_name="walk", task_kwargs={'random': r0})

# Helper function to calculate the total dimension of observation spaces
def calculate_total_dim(specs):
    """
    Calculates the total dimension of observation spaces by summing the products of each dimension.
    Args:
        specs: Observation space specifications.
    Returns:
        Total dimension as an integer.
    """
    return sum(np.prod(spec.shape) for spec in specs.values())

# Calculate observation and action dimensions
observation_spec = env.observation_spec()
observation_dim = int(calculate_total_dim(observation_spec))
action_spec = env.action_spec()
action_dim = np.prod(action_spec.shape)

# Define a wrapper for the environment to standardize the interface
class EnvironmentWrapper:
    """
    Wraps the environment to provide a standardized interface for observations and actions.
    """
    def __init__(self, env):
        self.env = env
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def reset(self):
        """
        Resets the environment and returns processed initial observations.
        """
        time_step = self.env.reset()
        return self._process_observations(time_step.observation)

    def step(self, action):
        """
        Steps through the environment using the given action and returns processed observations and other info.
        """
        time_step = self.env.step(action)
        observation = self._process_observations(time_step.observation)
        reward = time_step.reward
        done = time_step.last()
        return observation, reward, done, {}

    def _process_observations(self, observations):
        """
        Processes and flattens observations into a single array.
        """
        processed = np.concatenate([np.ravel(observations[key]) for key in observations])
        return processed

wrapped_env = EnvironmentWrapper(env)

# Initialize PPO with defined hyperparameters
hyperparameters = {
    'create_new_training_data': True,
    'load_model': False,
    'training_path': './training/walker/walk'
}

ppo_agent = PPO(wrapped_env, **hyperparameters)

# Load the previously trained model and set it to evaluation mode
ppo_agent.ac_model.load_state_dict(torch.load('./training/walker/walk/best_ppo_ac_model - 915.pth'))
ppo_agent.ac_model.eval()

# Define the policy function for use with the DM Control Viewer
def policy(time_step):
    """
    Defines a policy function that computes the action to take at each time step.
    """
    if time_step.first():
        action = np.zeros(env.action_spec().shape)  # Initialize with zeros
    else:
        observation = wrapped_env._process_observations(time_step.observation)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action_tensor, _, _ = ppo_agent.ac_model.step(obs_tensor)
        if isinstance(action_tensor, np.ndarray):
            action = action_tensor
        else:
            action = action_tensor.numpy()  # Convert to numpy array if necessary
    return action

# Launch the viewer to visualize the policy in action
viewer.launch(env, policy)
