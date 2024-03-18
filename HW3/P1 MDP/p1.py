import numpy as np
import matplotlib.pyplot as plt


def generate_transition_matrix(env_shape, obstacles_list, goal_idx):
    # The environment's shape is a square grid of size 'n'
    n = env_shape[0]
    
    # Initialize the transition matrix with zeros
    # The matrix is 5-dimensional: state (i, j), action, resulting state (new_i, new_j)
    # Actions are encoded as 0: left, 1: right, 2: up, 3: down
    T = np.zeros((n, n, 4, n, n))
    
    # Iterate over all cells in the grid to set transition probabilities
    for i in range(n):
        for j in range(n):
            # If the cell is the goal state, any action results in staying in the goal with probability 1
            if (i, j) == goal_idx:
                T[i, j, :, i, j] = 1
                continue  # Skip further processing for the goal state

            # If the cell is an obstacle, skip it as no action is applicable
            if (i, j) in obstacles_list:
                continue
            
            # Define transitions based on the control action taken by the robot
            # For each direction, there's a primary movement direction with p=0.7
            # and secondary movements (including staying in place) with p=0.1
            
            # For action 0 (left)
            T[i, j, 0, i, np.clip(j-1, 0, n-1)] += 0.7
            T[i, j, 0, np.clip(i-1, 0, n-1), j] += 0.1
            T[i, j, 0, np.clip(i+1, 0, n-1), j] += 0.1
            T[i, j, 0, i, j] += 0.1
            
            # For action 1 (right)
            T[i, j, 1, i, np.clip(j+1, 0, n-1)] += 0.7
            T[i, j, 1, np.clip(i-1, 0, n-1), j] += 0.1
            T[i, j, 1, np.clip(i+1, 0, n-1), j] += 0.1
            T[i, j, 1, i, j] += 0.1
            
            # For action 2 (up)
            T[i, j, 2, np.clip(i-1, 0, n-1), j] += 0.7
            T[i, j, 2, i, np.clip(j+1, 0, n-1)] += 0.1
            T[i, j, 2, i, np.clip(j-1, 0, n-1)] += 0.1
            T[i, j, 2, i, j] += 0.1
            
            # For action 3 (down)
            T[i, j, 3, np.clip(i+1, 0, n-1), j] += 0.7
            T[i, j, 3, i, np.clip(j+1, 0, n-1)] += 0.1
            T[i, j, 3, i, np.clip(j-1, 0, n-1)] += 0.1
            T[i, j, 3, i, j] += 0.1

    return T


def generate_state_map(env_shape):
    """
    Generates a map of the environment with obstacles and points of interest.

    Args:
        env_shape (tuple): The dimensions of the environment (height, width).
    
    Returns:
        np.ndarray: A 2D array representing the environment where
                    0 = free space,
                    1 = obstacle,
                    2 = goal,
                    3 = initial position,
                    4 = point of interest.
    """
    # Initialize the environment with free spaces
    state_map = np.zeros(env_shape)
    
    # Set the boundaries of the environment as obstacles
    state_map[:, 0] = state_map[:, -1] = state_map[0, :] = state_map[-1, :] = 1
    
    # Add specific obstacles within the environment
    state_map[2, 3:7] = 1
    state_map[4:8, 4] = 1
    state_map[7, 5] = 1
    state_map[4:6, 7] = 1

    # Flip the map if needed to match the desired orientation
    # state_map = np.flip(state_map, axis=1)
    # state_map = np.flip(state_map)

    # Mark specific points of interest
    state_map[8, 8] = 2  # Goal position
    state_map[8, 1] = 3  # Initial position
    state_map[3, 3] = 4  # Another point of interest

    return state_map


def generate_Q_map(states, env_shape, reward):
    """
    Generate the Q-value map for an environment.
    
    Args:
        states (np.ndarray): A 2D array where cells are marked with 0 for free space,
                             1 for obstacles, and 2 for the goal.
        env_shape (tuple): Shape of the environment as (height, width).
        reward (float): The reward for reaching the goal or hitting an obstacle.
        
    Returns:
        np.ndarray: A 3D array representing the Q-values for each action at each cell.
                    The shape of the array is (height, width, 4), corresponding to
                    the environment dimensions and four possible actions.
    """
    
    # Identify the coordinates of obstacles and the goal in the grid
    obstacles = np.where(states == 1)
    goal_idx = np.where(states == 2)

    # Initialize the Q-value map with -1 for all states and actions
    # This encourages exploration by giving a slightly negative value to unvisited states
    Q = -1 * np.ones((env_shape[1], env_shape[0], 4))  # Notice the corrected order of dimensions
    
    # Assign a negative reward to all actions leading to obstacle states
    # This penalizes hitting obstacles
    Q[obstacles[0], obstacles[1], :] = -reward
    
    # Assign a positive reward to all actions leading to the goal state
    # This incentivizes reaching the goal
    Q[goal_idx[0], goal_idx[1], :] = reward

    return Q



def policy_evaluation(T, Q, J_init, u):
    """
   Evaluates a policy to estimate the state-value function for each state.
   
   J[k, i, j]: This represents the estimated value (utility) of being in state (i, j) at iteration k under a specific policy u. The value function J estimates the total expected rewards from being in a particular state and following a certain policy thereafter.

    Q[i, j, u[i, j]]: This is the immediate reward (or the action-value) of taking action u[i, j] (the action recommended by the policy at state (i, j)) plus the expected future rewards. The Q matrix stores the action-value function, which gives the quality of each action at each state.

    gamma: This is the discount factor (denoted as γ). It represents the difference in importance 
    
    J[k-1].flatten().T: This term represents the value function from the previous iteration (k-1)

   Args:
       T (np.ndarray): The transition probabilities matrix of shape (n, n, 4, n, n).
       Q (np.ndarray): The action-value function matrix of shape (env_height, env_width, actions).
       J_init (np.ndarray): Initial state-value function of shape (env_height, env_width).
       u (np.ndarray): Policy matrix indicating the action for each state.
   
   Returns:
       np.ndarray: Estimated state-value function after policy evaluation.
   """
    iter = 300
    J = np.zeros((iter, env_shape[0], env_shape[1]))
    J[0] = J_init
    for k in range(1, iter):
        for i in range(10):
            for j in range(10):
                J[k,i,j] = Q[i,j,u[i,j]] + gamma*(T[i,j,u[i,j]].flatten() @ J[k-1].flatten().T)

    return J[-1]




def policy_improvement(T, Q, J_new, gamma, env_shape):
    """
    Performs policy improvement by finding an improved policy based on the updated state-value function.

    Args:
        T (np.ndarray): Transition probabilities matrix of shape (env_height, env_width, actions, env_height, env_width).
        Q (np.ndarray): Action-value function matrix of shape (env_height, env_width, actions).
        J_new (np.ndarray): Updated state-value function of shape (env_height, env_width) from policy evaluation.
        gamma (float): Discount factor for future rewards.
        env_shape (tuple): Shape of the environment (height, width).
    
    Returns:
        np.ndarray: Improved policy matrix indicating the best action for each state.
    """
    # Define a lambda function to find the action that maximizes the expected utility for each state
    get_best_action = lambda action_values: np.argmax(action_values)

    # Initialize the improved policy matrix with zeros
    u_k = np.zeros(env_shape)

    # Iterate over all states in the environment
    for i in range(env_shape[0]):
        for j in range(env_shape[1]):
            # Extract Q-values and transition probabilities for the current state
            Q_element = Q[i, j]
            T_element = T[i, j].reshape(4, -1)
            
            # Compute the expected utility for each action and select the best action
            u_k[i, j] = get_best_action(Q_element + (gamma * T_element @ J_new.reshape((-1, 1))).reshape(4))

    # Return the improved policy
    return u_k

def visualize(J_new, u, obstacles_list, goal_idx_list, iteration_number):
    """
    Visualizes the policy and value function of a grid.

    Args:
        J_new (np.ndarray): The value function to be visualized.
        u (np.ndarray): The policy matrix, with actions for each cell.
        obstacles_list (list of tuples): Coordinates of obstacles in the grid.
        goal_idx_list (list of tuples): Coordinates of goal(s) in the grid.
    """
      
    # Set up the figure and axes for the visualization
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks(np.arange(0.5, 10.5, 1))
    ax.set_yticks(np.arange(0.5, 10.5, 1))
    
    # Use a heatmap to visualize the value function
    cmap = plt.get_cmap('bwr') 
    im = plt.imshow(J_new, cmap=cmap)
    plt.imshow(J_new, cmap=cmap)
    plt.grid(color='gray', linestyle='--', linewidth=0.5) 
    
    # Draw arrows to represent the policy at each cell
    for i in range(10):
        for j in range(10):
            if (i, j) not in obstacles_list and (i, j) not in goal_idx_list:
                dx, dy = 0, 0
                if u[i, j] == 0:  # Left
                    dx = -0.25
                elif u[i, j] == 1:  # Right
                    dx = 0.25
                elif u[i, j] == 2:  # Up
                    dy = -0.25
                elif u[i, j] == 3:  # Down
                    dy = 0.25
                ax.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')  # Change arrow color here
    
    fig.colorbar(im, ax=ax)
    plt.title(f"Actions taken at the {iteration_number} iteration(s)")
    plt.show()


# Initial setup
env_shape = (10, 10)
reward = 10
gamma = 0.9

# Generate environment states, transition matrix, and initial Q-values
states = generate_state_map(env_shape)
obstacles_list = list(zip(*np.where(states == 1)))
goal_idx_list = list(zip(*np.where(states == 2)))
T = generate_transition_matrix(env_shape, obstacles_list, goal_idx_list[0])
Q = generate_Q_map(states, env_shape, reward)

# Initialize policy evaluation and improvement
num_policy_iter = 4
J = np.zeros(env_shape)
u = np.ones((num_policy_iter+1, env_shape[0], env_shape[1]), dtype=int)

# Iterate over policy evaluation and improvement
for k in range(num_policy_iter):
    J_new = policy_evaluation(T, Q, J, u[k])
    J = J_new
    u[k+1] = policy_improvement(T, Q, J_new, gamma, env_shape)
    visualize(J_new, u[k+1],obstacles_list, goal_idx_list,k)

















