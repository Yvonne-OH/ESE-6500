import numpy as np
import scipy.signal
import torch

def discount_cumsum(x, discount):
    """
    Function adapted from rllab for computing discounted cumulative sums of vectors.
    Input:
        vector x, example:
        [x0, x1, x2]
    Output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO (Proximal Policy Optimization) agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.mean_rews = []

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        Ensures that there is room in the buffer to store the data.
        """
        assert self.ptr < self.max_size  # buffer must have room to store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        This method should be called at the end of a trajectory, or when it is cut off by an epoch ending.
        It looks back in the buffer to the start of the trajectory and uses the rewards and value estimates
        from the entire trajectory to compute advantage estimates with GAE-Lambda and the rewards-to-go for each state.
        The `last_val` argument should be 0 if the trajectory ended because the agent reached a terminal state (died),
        otherwise it should be the estimated value function V(s_T) for the last state.
        This allows for bootstrapping the reward-to-go calculation to account for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        self.mean_rews.append(np.mean(rews))

        # Compute GAE-Lambda for advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # Compute rewards-to-go, which are targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this method at the end of an epoch to retrieve all data from the buffer,
        with advantages normalized to have mean zero and standard deviation one.
        This method also resets some internal pointers in the buffer.
        """
        assert self.ptr == self.max_size  # ensure buffer is full before retrieving data
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        # Return data as a dictionary and convert to torch tensors
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in dict(
            obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf, mean_rews=self.mean_rews).items()}

        # Reset the mean_rews list
        self.mean_rews = []
        return data
