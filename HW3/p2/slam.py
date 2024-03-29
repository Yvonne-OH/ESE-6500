import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))


class map_t:
    """
    This will maintain the occupancy grid and log_odds.
    You do not need to change anything in the initialization
    """

    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szy = int(np.ceil((s.ymax - s.ymin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)  # initialize the map as empty
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding to these (x,y) locations.
        You should return an array of shape (2 x len(x)).
        Be careful to handle instances when x/y go outside the map bounds,
        you can use np.clip to handle these situations.
        """
        #### TODO: Checked
        x_indices = np.clip((x - s.xmin) / s.resolution, 0, s.szx - 1).astype(int)
        y_indices = np.clip((y - s.ymin) / s.resolution, 0, s.szy - 1).astype(int)
        
        # Stack the x_indices and y_indices vertically to create a 2D array where each column represents
        return np.vstack((x_indices, y_indices))


class slam_t:
    """
    s is the same as s. In Python it does not really matter
    what we call s, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """

    def __init__(s, resolution=0.05, Q=1e-3 * np.eye(3), resampling_threshold=0.3):

        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = Q
        # s.Q = 1e-8 * np.eye(3)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir, 'data/%s/%s_lidar%d' % (split, split, idx)))
        s.joint = load_joint_data(os.path.join(src_dir, 'data/%s/%s_joint%d' % (split, split, idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t'] - t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25  # degrees
        s.lidar_angles = np.arange(-135, 135 + s.lidar_angular_resolution,
                                      s.lidar_angular_resolution) * np.pi / 180.0
        # Sensor model
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1 / 9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n) / float(s.n)  # 1/n

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        
        Parameters:
        - p: (3 x n) numpy array of particle states.
        - w: (1 x n) numpy array of particle weights.
        
        Returns:
        - Tuple of (resampled_particles, uniform_weights):
          - resampled_particles is a (3 x n) numpy array after resampling.
          - uniform_weights is a (1 x n) numpy array of equal weights for all particles.
        """
        #### TODO: Checked
        n = len(w)  # Total number of particles
        # Adjust weights and repeat particles based on adjusted weights
        adjusted_weights = (w * n * 10).astype(int)
        repeated_particles = np.repeat(p, adjusted_weights, axis=1)
        
        # Select a subset of particles randomly
        indexes = np.random.choice(repeated_particles.shape[1], n, replace=False)
        resampled_particles = repeated_particles[:, indexes]
        
        # Assign equal weight to all resampled particles
        uniform_weights = np.ones(n) / n
        
        return resampled_particles, uniform_weights


    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w - w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p: p is the pose of the particle (x,y,yaw), a 3x1 array describing the robot position and orientation
        d: an array that stores the distance along the ray of the lidar for each ray
           the length of d has to be equal to that of angles, this is s.lidar[t]['scan']
        head_angle: the angle of the head in the body frame, usually 0, need to be in radians
        neck_angle: the angle of the neck in the body frame, usually 0, need to be in radians
        angles: angle of each ray in the body frame in radians
                (usually be simply s.lidar_angles for the different lidar rays)

        Return an array (2 x num_rays) which are the (x,y) locations of the end point of each ray in world coordinates
        """
        # Filter valid LiDAR points based on distance constraints
        in_range = np.logical_and(d >= s.lidar_dmin, d <= s.lidar_dmax)
        d_filtered = d[in_range]
        angle_filtered = angles[in_range]
        
        # Transform distances to LiDAR frame points (2D to 3D)
        lidar_pts = np.vstack((d_filtered * np.cos(angle_filtered), d_filtered * np.sin(angle_filtered), np.zeros(d_filtered.size)))
        
        # Transformation from LiDAR to body frame
        lidar_to_body_tf = euler_to_se3(0, head_angle, neck_angle, np.array([0, 0, s.lidar_height]))
        body_pts_4d = lidar_to_body_tf @ make_homogeneous_coords_3d(lidar_pts)  # Transform to 4D for matrix multiplication
        
        # Transform from body frame to world frame
        body_to_world_tf = euler_to_se3(0, 0, p[2, 0], np.array([p[0, 0], p[1, 0], s.head_height]))
        world_pts_4d = body_to_world_tf @ body_pts_4d  # Apply transformation
        
        # Normalize and return 2D world frame points
        world_pts_2d = world_pts_4d[:3] / world_pts_4d[3]
        return world_pts_2d[:2]

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """
        
        """    
                
        Parameters:
        - t: The current time step (index) in the LIDAR data sequence.
        
        Returns:
        - control: The computed control signal as a difference in x, y coordinates,
                   and heading angle (theta),
                   indicating how to adjust the pose from time step t-1 to t.
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: Checked
        
        # Compute the difference in pose between time t and t-1
        # Extract the previous and current poses from LIDAR data using the given time step t.
        previous_pose = s.lidar[t - 1]['xyth']
        current_pose = s.lidar[t]['xyth']

       # Compute the control signal as the difference between current and previous poses.
        return smart_minus_2d(current_pose, previous_pose)

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        
        """
        Parameters:
            - control: The control signal to be applied, typically the difference in pose.
        """
        #### TODO: Checked
        control = s.get_control(t)

        # Generate noise for all particles at once
        noise = np.random.multivariate_normal(np.zeros(s.Q.shape[0]), s.Q, s.n)
        
        # Apply the noisy control to each particle
        for i in range(s.n):
            noisy_control = control + noise[i]
            s.p[:, i] = smart_plus_2d(s.p[:, i].copy(), noisy_control)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
          new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        # Parse the observation log-probability
        w = obs_logp + np.log(w)
        w -= slam_t.log_sum_exp(w)
        w = np.exp(w)
        return w

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
    
        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: checked
        # Extract head and neck angles
        idx = s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])
        angle_neck = s.joint['head_angles'][0, idx]
        angle_head = s.joint['head_angles'][1, idx]
    
        # Initialize observation probabilities
        log_prob_obs = np.zeros(s.n)
        
        for i in range(s.n):
            # Project lidar scan ---> world frame
            p = s.p[:, i].reshape((3, 1))
            world_frame_points = s.rays2world(p, s.lidar[t]['scan'], angle_head, angle_neck, s.lidar_angles)
    
            # grid cell indices of the occupied cells && observation log-probability
            occupied_cells = s.map.grid_cell_from_xy(world_frame_points[0], world_frame_points[1])
            log_prob_obs[i] = np.sum(s.map.log_odds[occupied_cells[0], occupied_cells[1]])
    
        # Update particle weights and estimate the pose from the best particle
        s.w = s.update_weights(s.w, log_prob_obs)
        best_idx = np.argmax(s.w)
        s.estimated_pose = s.p[:, best_idx]
    
        # Update the map based on the best particle's observation
        best_particle_world = s.rays2world(s.estimated_pose.reshape((3, 1)), s.lidar[t]['scan'], angle_head, angle_neck, s.lidar_angles)
        occupied_x, occupied_y = s.map.grid_cell_from_xy(best_particle_world[0], best_particle_world[1])
    
        # Compute and update free cells from best particle to observed obstacles
        limits_x, limits_y = s.calculate_free_space(s.estimated_pose, occupied_x, occupied_y)
        s.update_map(occupied_x, occupied_y, limits_x, limits_y)
    
        # Resample particles 
        s.resample_particles()
        
    def calculate_free_space(s, pose, occupied_x, occupied_y):
        """
        Calculate free space coordinates based on the current pose and observed occupied cells.
        """
        # Compute limits based on lidar maximum distance and current pose
        limit_x = np.array([pose[0] - s.lidar_dmax / 2, pose[0] + s.lidar_dmax / 2, pose[0]])
        limit_y = np.array([pose[1] - s.lidar_dmax / 2, pose[1] + s.lidar_dmax / 2, pose[1]])
        limit_grid_x, limit_grid_y = s.map.grid_cell_from_xy(limit_x, limit_y)
    
        # Determine free cells
        free_x = np.linspace(limit_grid_x[2], occupied_x, endpoint=False).astype(int).flatten()
        free_y = np.linspace(limit_grid_y[2], occupied_y, endpoint=False).astype(int).flatten()
    
        return free_x, free_y
    
    
    def update_map(s, occupied_x, occupied_y, free_x, free_y):
        """
        Update SLAM map log-odds values based on observed and free cells, then binarize the map.
        """
        # Update log-odds for occupied and free cells
        s.map.log_odds[occupied_x, occupied_y] += s.lidar_log_odds_occ
        s.map.log_odds[free_x, free_y] += s.lidar_log_odds_free
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)
    
        # Binarize the map based on log-odds threshold
        s.map.cells = (s.map.log_odds > s.map.log_odds_thresh).astype(int)

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance in the particles.
        We should resample only if the effective number of particles falls below
          a certain threshold (resampling_threshold).
        A good heuristic to calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
          of the particles, if this number of close to n, then all particles have about equal weights,
          and we do not need to resample
        """
        e = 1 / np.sum(s.w ** 2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e / s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
