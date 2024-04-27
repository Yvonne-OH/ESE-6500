import numpy as np
import torch
from dm_control import suite
from RL_brain import PPO  # Make sure your PPO class and other necessary classes are properly defined in RL_brain.py

# Load the environment
r0 = np.random.RandomState(42)
env = suite.load(domain_name="walker", task_name="walk", task_kwargs={'random': r0})

# Function to flatten observations
def flatten_observation(obs_dict):
    flat_obs = []
    for key in obs_dict:
        flat_obs.append(obs_dict[key].ravel())
    return np.concatenate(flat_obs, axis=0)

# Determine the size of the flattened observation
example_observation = env.reset().observation
flattened_observation = flatten_observation(example_observation)
n_states = flattened_observation.shape[0]
n_actions = env.action_spec().shape[0]

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
n_hiddens = 64
actor_lr = 1e-3
critic_lr = 1e-3
lmbda = 0.95
epochs = 10
eps = 0.2
gamma = 0.99

# Initialize PPO
ppo = PPO(n_states, n_hiddens, n_actions, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

# Training loop parameters
num_episodes = 1000
max_steps = 1000
log_interval = 10  # Log and print every 10 episodes

# Training loop
# Training loop
for episode in range(num_episodes):
    time_step = env.reset()
    state = flatten_observation(time_step.observation)
    total_reward = 0
    transitions = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': []
    }

    for step in range(max_steps):
        action = ppo.take_action(state)
        next_time_step = env.step(action)  # Assuming the environment accepts the action array directly
        next_state = flatten_observation(next_time_step.observation)
        reward = next_time_step.reward
        done = next_time_step.last()

        transitions['states'].append(state)
        transitions['actions'].append(action)
        transitions['rewards'].append(reward)
        transitions['next_states'].append(next_state)
        transitions['dones'].append(float(done))

        state = next_state
        total_reward += reward

        if done:
            break

    # Convert lists to numpy arrays
    for key in transitions:
        transitions[key] = np.array(transitions[key])

    # Update PPO
    ppo.update(transitions)

    # Logging
    if episode % log_interval == 0:
        print(f'Episode {episode}: Total Reward = {total_reward}')
