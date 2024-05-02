import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Creates a multilayer perceptron (MLP) with the specified layer sizes,
    activations for each layer, and output activation.
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation[j] if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
  
class MLPCritic(nn.Module):
    
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        # Initializes a critic neural network to estimate the value function.
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Forward pass to compute the value function for a given observation.
        # Ensures output has the correct shape by squeezing last dimension.
        return torch.squeeze(self.v_net(obs), -1)       

class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # Initializes an actor neural network to estimate the mean of action distributions.
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        # Creates a Normal distribution with mean `mu` and standard deviation `std`.
        return Normal(mu, std)

    def _log_prob_from_distribution(self, dist, act):
        # Calculates the log probability of `act` under the distribution `dist`.
        return dist.log_prob(act).sum(axis=-1)    # Summing over the last axis for log probabilities.  

    def forward(self, obs, act=None):
        # Forward pass to produce action distributions for given observations and
        # optionally compute the log likelihood of given actions under those distributions.
        dist = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(dist, act)
        return dist, logp_a              


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128,128), activation=nn.Tanh):
        super().__init__()
        # Initializes actor-critic architecture with policy (actor) and value function (critic).
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.vf = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        # Compute action, value function, and log probability for a given observation.
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            a = dist.sample()
            logp_a = self.pi._log_prob_from_distribution(dist, a)
            v = self.vf(obs)
        # Returns action, value, and log probability as NumPy arrays without gradient information.
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        # Extracts only the action part of the step method.
        return self.step(obs)[0]
