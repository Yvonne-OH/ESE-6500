"""
Created on Sun Apr 28 15:25:41 2024
@author: Zhanqian Wu
"""

from policies import MLPActorCritic
from buffer import PPOBuffer
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

class PPO():
    def __init__(self, env, **hyperparameters):
        """
        Initializes the PPO class with the environment and a set of hyperparameters.
        """
        # Update hyperparameters based on user input
        self.set_hyperparameters(hyperparameters)
        
        # Retrieve dimensions of the environment's observation and action spaces
        self.env = env
        self.obs_dim = self.env.observation_dim
        self.act_dim = self.env.action_dim

        # Create neural network model for both actor and critic
        self.ac_model = MLPActorCritic(self.obs_dim, self.act_dim, self.hidden, self.activation)

        # Set up optimizers for the actor and critic
        self.pi_optimizer = Adam(self.ac_model.pi.parameters(), self.pi_lr)
        self.vf_optimizer = Adam(self.ac_model.vf.parameters(), self.vf_lr)

        # Buffer for storing trajectories
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.gamma, self.lam)

        # Logging data
        self.logger = {'mean_rew': 0, 'std_rew': 0}
      
        # Create directory for saving training data and model if it doesn't exist
        if not os.path.exists(self.training_path):
            os.makedirs(self.training_path)
            print(f"New directory created: {self.training_path}")
        
        # Prepare CSV file for logging training data
        self.column_names = ['mean', 'std']
        self.df = pd.DataFrame(columns=self.column_names, dtype=object)
        if self.create_new_training_data:
            self.df.to_csv(os.path.join(self.training_path, self.data_filename), mode='w', index=False)  
            print(f"New data file created: {self.data_filename}")            
        
        # Load model from file if specified
        if self.load_model:
            self.ac_model.load_state_dict(torch.load(os.path.join(self.training_path, self.model_filename)))
            print(f"Model loaded: {self.model_filename}")
    
    def set_hyperparameters(self, hyperparameters):
        """
        Set default values for hyperparameters and update them based on provided inputs.
        """
        self.epochs = 8000
        self.steps_per_epoch = 1000
        self.max_ep_len = 1000
        self.gamma = 0.99
        self.lam = 0.97
        self.clip_ratio = 0.06
        self.target_kl = 0.01
        self.coef_ent = 0.001
        self.train_pi_iters = 20
        self.train_vf_iters = 20
        self.pi_lr = 3e-5
        self.vf_lr = 1e-4
        self.hidden = (64, 64)
        self.activation = [nn.Tanh, nn.ReLU]
        self.flag_render = False
        self.save_freq = 100
        self.training_path = './training/hopper/standUp'
        self.data_filename = 'data'
        self.model_filename = 'ppo_ac_model.pth'
        self.create_new_training_data = False
        self.load_model = False        

        # Update default hyperparameters 
        for param, val in hyperparameters.items():
            exec("self." + param + "=" + "val")

    def compute_loss_pi(self, data):
        """
        Calculate the policy loss using provided data.
        """
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        act_dist, logp = self.ac_model.pi(obs, act)  # Evaluate new policy
        ratio = torch.exp(logp - logp_old)  # Importance sampling ratio
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        ent = act_dist.entropy().mean().item()  # Entropy bonus
        loss_pi = -(torch.min(ratio * adv, clip_adv) + self.coef_ent * ent).mean()
        approx_kl = (logp_old - logp).mean().item()  # Approximate KL-divergence
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info
    
    def compute_loss_vf(self, data):
        """
        Calculate the value function loss using provided data.
        """
        obs, ret = data['obs'], data['ret']
        return ((self.ac_model.vf(obs) - ret)**2).mean()
    
    def update(self):
        """
        Perform a policy update using the collected data from the buffer.
        """
        data = self.buf.get()
        self.logger['mean_rew'] = data['mean_rews'].mean().item()
        self.logger['std_rew'] = data['mean_rews'].std().item()
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 2.5 * self.target_kl:
                print(f"Early stopping at step {i} due to max kl")
                break
            loss_pi.backward()
            self.pi_optimizer.step()
        for i in range(self.train_vf_iters):
            self.vf_optimizer.zero_grad()
            loss_vf = self.compute_loss_vf(data)
            loss_vf.backward()
            self.vf_optimizer.step()
    
    def rollout(self):
        """
        Generate data by interacting with the environment according to the current policy.
        """
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        total_epoch_reward = 0
        for t in range(self.steps_per_epoch):
            obs_tensor = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            a, v, logp = self.ac_model.step(obs_tensor)
            a = a[0]  # Remove batch dimension
            next_o, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1
            total_epoch_reward += r
            self.buf.store(o, a, r, v, logp)
            o = next_o
            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t == self.steps_per_epoch - 1
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                if timeout or epoch_ended:
                    _, v, _ = self.ac_model.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                else:
                    v = 0
                self.buf.finish_path(v)
                o, ep_ret, ep_len = self.env.reset(), 0, 0
        return total_epoch_reward
    
    def adjust_learning_rate(self, optimizer, new_lr):
        """
        Update the learning rate of the specified optimizer.
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def learn(self):
        """
        Execute the learning process for the specified number of epochs.
        """
        best_reward = -float('inf')
        mean_rewards = []
        for epoch in range(self.epochs):
            total_reward = self.rollout()
            self.update()
            mean_rewards.append(total_reward)
            if total_reward > best_reward:
                best_reward = total_reward
                best_model_path = os.path.join(self.training_path, "best_" + self.model_filename)
                torch.save(self.ac_model.state_dict(), best_model_path)
                print(f"New best model saved with reward: {best_reward}")
            if (epoch + 1) % 10 == 0:
                print("====================")
                print(f"Epoch: {epoch + 1}")
                print(f"Total Reward: {total_reward}")
                print("====================\n")
                plt.figure()
                plt.plot(mean_rewards, label='Total Reward')
                plt.title('Training Progress - Total Reward')
                plt.xlabel('Epochs')
                plt.ylabel('Total Reward')
                plt.legend()
                plt.grid(True)
                plt.show()
            if (epoch + 1) % self.save_freq == 0:
                torch.save(self.ac_model.state_dict(), os.path.join(self.training_path, self.model_filename))
                print(f"Model saved at epoch {epoch + 1}.")
                rewards_df = pd.DataFrame(mean_rewards, columns=['Total Reward'])
                rewards_df.to_csv(os.path.join(self.training_path, 'training_log.csv'), index_label='Epoch')

if __name__=='__main__':
    r0 = np.random.RandomState(42)
    env = suite.load(domain_name="walker", task_name="walk", task_kwargs={'random': r0})
    
    def calculate_total_dim(specs):
        """
        Calculate the total dimension of the observation space by summing the product of each dimension.
        """
        return sum(np.prod(spec.shape) for spec in specs.values())
    
    observation_spec = env.observation_spec()
    observation_dim = int(calculate_total_dim(observation_spec))
    action_spec = env.action_spec()
    action_dim = np.prod(action_spec.shape)
    
    class EnvironmentWrapper:
        """
        Wraps the environment to provide a standardized interface.
        """
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
            """
            Convert and flatten observations into a single array.
            """
            processed = np.concatenate([np.ravel(observations[key]) for key in observations])
            return processed
    
    wrapped_env = EnvironmentWrapper(env)
    
    hyperparameters = {   
        'create_new_training_data': True,
        'load_model': False,
        'training_path': './training/walker/walk'
    }
    
    ppo_agent = PPO(wrapped_env, **hyperparameters)
    ppo_agent.learn()

