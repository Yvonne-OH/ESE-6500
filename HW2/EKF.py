import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# Parameters
a_true = -1  # True value of a
x0_mean = 1
x0_var = 2
eps_var = 1
nu_var = 1/2
num_observations = 100

# Initialization
np.random.seed(12)
x0 = np.random.normal(1, np.sqrt(2))
x = np.zeros(num_observations)
y = np.zeros(num_observations)
x[0] = x0

# Simulation
for k in range(1, num_observations):
    epsilon_k = np.random.normal(0, np.sqrt(eps_var))
    x[k] = a_true * x[k-1] + epsilon_k
    nu_k = np.random.normal(0, np.sqrt(nu_var))
    y[k] = np.sqrt(x[k]**2 + 1) + nu_k



def ekf_update_fixed(x_est, P, y, Q, R):
    """
    Extended Kalman Filter update.
    
    Args:
    x_est: Estimated state vector [x_k, a] at time k-1.
    P: Estimated state covariance matrix at time k-1.
    y: Observation at time k.
    Q: Process noise covariance matrix.
    R: Observation noise covariance matrix.
    
    Returns:
    x_est_new: Updated state estimate at time k.
    P_new: Updated state covariance matrix at time k.
    """
    
    # Prediction step
    F = np.array([[x_est[1], x_est[0]], [0, 1]]) # Jacobian of the state transition 
    x_pred = np.array([x_est[1]*x_est[0], x_est[1]]) # Predicted state
    P_pred = F @ P @ F.T + Q # Predicted covariance
    
    # Update step
    x_temp = x_pred[0] # Temporary variable for readability
    H = np.array([x_temp/np.sqrt(x_temp**2 + 1), 0]).reshape(1, -1) # Jacobian of the observation model reshaped for scalar observation
    y_pred = np.sqrt(x_temp**2 + 1)                 # Predicted observation
    S = H @ P_pred @ H.T + R                        # Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)             # Kalman gain
    x_est_new = x_pred + K @ (y - y_pred)           # Updated state estimate. 
    P_new = (np.eye(2) - K @ H) @ P_pred            # Updated covariance estimate
    
    return x_est_new, P_new



y_series = y # Given a series of observation values y
x_series = x

x_est = np.array([x0, -5])              # Initial state estimate [x_0, a]
P = np.eye(2)                           # Initial state covariance
Q = np.array([[1,0],[0,0.5]])             # Process noise covariance 
R = np.array([np.sqrt(0.5)])            # Observation noise covariance

# Update EKF for each observation value
estimates = []
uncertainties = []
x_estimates = []
x_uncertainties = []
for y in y_series:
    x_est, P_new = ekf_update_fixed(x_est, P, np.array([y]), Q, R)
    
    estimates.append(x_est[1])
    uncertainties.append((P_new[1, 1]))
    
    x_estimates.append(x_est[0])
    x_uncertainties.append((P_new[0, 0]))
    
time_k = np.arange(num_observations)
plt.figure(figsize=(10, 6))
plt.plot(time_k, [a_true]*num_observations, label='True value of a', color='r')
plt.plot(time_k, estimates, label='Estimated value of a', color='b')
plt.fill_between(time_k, np.array(estimates) - np.array(uncertainties), np.array(estimates) + np.array(uncertainties), color='b', alpha=0.2, label='Confidence interval (±σ)')
plt.xlabel('Time step (k)')
plt.ylabel('Value')
plt.title('True and Estimated Values of a Over Time')
plt.legend()
plt.grid(True)
plt.show()


time_k = np.arange(num_observations)
plt.figure(figsize=(10, 6))
plt.plot(time_k, y_series, label='Dataset (observation)', color='r')
plt.plot(time_k, x_estimates, label='x estimates', color='b')
plt.plot(time_k, x_series,'k--',label='x ground truth')
plt.fill_between(time_k, np.array(x_estimates) - np.array(x_uncertainties), np.array(x_estimates) + np.array(x_uncertainties), color='b', alpha=0.2, label='Confidence interval (±σ)')
plt.xlabel('Time step (k)')
plt.ylabel('Value')
plt.title('True and Estimated Values of x and y Over Time')
plt.legend()
plt.grid(True)
plt.show()