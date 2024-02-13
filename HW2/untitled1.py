# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:29:19 2024

@author: 39829
"""

import numpy as np
from scipy import io
from scipy import linalg
# from scipy.spatial import transform
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def load_imu_data(data_num):
    # imu: dict of data with keys 'ts', 'vals' and respective values:
    # (1x5645) array of timestamps | (6x5645) array of measurements
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat') 
    accel = imu['vals'][0:3, :] # (3, 5645) array of ADC ints
    # for gyro we're given w_z, w_x, w_y ordering!!! so convert to x,y,z below
    gyro  = np.vstack((imu['vals'][4, :], imu['vals'][5, :], imu['vals'][3, :])) # (3, 5645) array of ADC ints
    T = np.shape(imu['ts'])[1]  # number of timesteps = 5645
    ts = imu['ts'][0] - imu['ts'][0][0] # (5645,) array of timestamps
    return accel, gyro, T, ts

def load_vicon_data(data_num):
    # Load "ground truth" VICON data (COMMENT OUT FOR AUTOGRADER SUBMISSION)
    # dict with keys 'rots' and 'ts' := 
    # (3, 3, 5561) array of rotations | (1x5561) array of timestamps
    vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    tv = vicon['ts'][0] - vicon['ts'][0][0] # (5645,) array of timestamps
    return vicon, tv

def discard_outliers(array, deviations = 3):
    #takes in a (n, m) array and returns a (n, m') array where m' <= m
    mean = np.mean(array, axis=1).reshape(array.shape[0],1)
    std = np.std(array, axis=1).reshape(array.shape[0],1)
    no_outliers = np.any(abs(array - mean) < deviations*std, axis=0)
    # print(f'removed {array.shape[1] - np.sum(no_outliers)} outliers')
    return array[:, no_outliers]

def auto_cal_window(accel, gyro, step = 50):
    # Compute bias and sensitivity for accelerometer and gyroscope
    # Output: accel_bias, accel_sensitivity, gyro_bias, gyro_sensitivity
    # baseline = np.linalg.norm(np.mean(accel[:, 0:step], axis=1))

    baseline = discard_outliers(accel[:, 0:step])
    base_mean = np.mean(baseline, axis=1)

    n = 1
    while n < (accel.shape[1] - step)/step:

        new = discard_outliers(accel[:, n*step:(n+1)*step])
        new_mean = np.mean(new, axis=1)
        base_std = np.std(baseline, axis=1)

        # print('norms: ', np.linalg.norm(new_mean), np.linalg.norm(base_mean))
        # print('stdev: ', np.std(baseline, axis=1))

        if np.linalg.norm(new_mean - base_mean) > np.maximum(np.linalg.norm(base_std), 0.1):
            break

        baseline = np.hstack((baseline, new))
        n += 1
        base_mean = ((n-1) * base_mean + new_mean) / n
    
    valid_cal_window = n * step
               
    return valid_cal_window

def ADCtoAccel(adc):
    '''
    Converts ADC readings from accelerometer to m/s^2
    Input:  adc - (int np.array shape (3, N)) ADC reading
    Output: acc - (float np.array shape (3, N)) acceleration in m/s^2
    '''
    bias        = np.array([510.808, 500.994, 499]).reshape(3,1)       # (mV)
    sensitivity = np.array([340.5, 340.5, 342.25]).reshape(3,1) # (mV/grav)
    return (adc.astype(np.float64) - bias) * 3300 / (1023 * sensitivity) * 9.81

def ADCtoGyro(adc, convert_to_rad=True):
    '''
    Converts ADC readings from gyroscope to rad/s
    Input:  adc - (int np.array shape (3, N)) ADC reading
    Output: gyr - (float np.array shape (3, N)) angular velocity in rad/s
    z,x,y ordering!!!
    '''
    bias        = np.array([373.568, 375.356, 369.68]).reshape(3,1)       # (mV)
    sensitivity = np.array([200, 200, 200]).reshape(3,1) # (mV/(rad/sec))
    if convert_to_rad:
        return (adc.astype(np.float64) - bias) * 3300 / (1023 * sensitivity) 
    else:
        return (adc.astype(np.float64) - bias) * 3300 / (1023 * sensitivity) * 180 / np.pi 

def VicontoRPY(vicon_rots):
    '''
    Converts Vicon rotation matrices to roll, pitch, yaw
    Input:  vicon - (float np.array shape (3, 3, N)) rotation matrices
    Output: roll  - (float np.array shape (N,)) roll angles in radians
    Output: pitch - (float np.array shape (N,)) pitch angles in radians
    Output: yaw   - (float np.array shape (N,)) yaw angles in radians
    '''
    N = vicon_rots.shape[2]
    roll  = np.zeros((N,))
    pitch = np.zeros((N,))
    yaw   = np.zeros((N,))

    for i in range(N):
        q = Quaternion()
        q.from_rotm(vicon_rots[:,:,i])
        roll[i], pitch[i], yaw[i] = q.euler_angles()
    return roll, pitch, yaw

def calibrate_imu(accel, gyro, cal_window):

    accel_bias        = np.array([510.808, 500.994, 499])
    accel_sensitivity = np.array([340.5, 340.5, 342.25])
    gyro_bias        = np.array([373.568, 375.356, 369.68])
    gyro_sensitivity = np.array([200, 200, 200])

    # accelerometer
    accel_bias = np.mean(accel[:, 0:cal_window], axis=1)
    mean_z = accel_bias[2]
    accel_bias[2] = np.mean(accel_bias[0:2])
    s = (mean_z - accel_bias[2]) * 3300 / 1023
    accel_sensitivity = np.ones((3,)) * s

    # gyro
    gyro_bias = np.mean(gyro[:, 0:cal_window], axis=1)
    gyro_sensitivity = np.array([200, 200, 200]) # (mV/(rad/sec))

    converted_accel = (accel.astype(np.float64) - accel_bias.reshape(3,1)) * 3300 / (1023 * accel_sensitivity.reshape(3,1)) * 9.81
    converted_gyro = (gyro.astype(np.float64) - gyro_bias.reshape(3,1)) * 3300 / (1023 * gyro_sensitivity.reshape(3,1))

    # print(f'biases: {accel_bias}, {gyro_bias}')
    # print(f'sensitivities: {accel_sensitivity}, {gyro_sensitivity}')

    return converted_accel, converted_gyro


def preprocess_dataset(data_num):
    # Load IMU data
    accel, gyro, nT, T_sensor = load_imu_data(data_num)

    cal_window = auto_cal_window(accel, gyro, step = 100)
    cal_time = T_sensor[cal_window]
    print(f'Calibration Window: {T_sensor[cal_window]}')

    calibrationPrint(accel, gyro, 'before transform')
    # Compute bias and sensitivity for accelerometer and gyroscope
    accel, gyro = calibrate_imu(accel, gyro, cal_window)
    
    # # Convert ADC readings to physical units
    # accel = ADCtoAccel(accel)
    # gyro  = ADCtoGyro(gyro)
    accel[0:2,:] *= -1 # flip readings per Warning
    calibrationPrint(accel, gyro, 'after transform')

    vicon, T_vicon = load_vicon_data(data_num)
    # Convert vicon rotation matrices to roll, pitch, yaw
    roll, pitch, yaw = VicontoRPY(vicon['rots'])

    # plotStuff(accel, gyro, T_sensor.ravel(), roll, pitch, yaw, T_vicon.ravel())

    return accel, gyro, T_sensor, roll, pitch, yaw, T_vicon, cal_time

def preprocess_dataset_no_vicon(data_num):
    # Load IMU data
    accel, gyro, nT, T_sensor = load_imu_data(data_num)

    cal_window = auto_cal_window(accel, gyro, step = 100)
    cal_time = T_sensor[cal_window]

    calibrationPrint(accel, gyro, 'before transform')

    # Convert ADC readings to physical units
    accel, gyro = calibrate_imu(accel, gyro, cal_window)
    # accel = ADCtoAccel(accel)
    # gyro  = ADCtoGyro(gyro)
    accel[0:2,:] *= -1 # flip readings per Warning

    calibrationPrint(accel, gyro, 'after transform')

    # plotStuff(accel, T_sensor.ravel(), roll, pitch, yaw, T_vicon.ravel())

    return accel, gyro, T_sensor

def plotStuff(accel, gyro, ts, roll, pitch, yaw, tv):
    plt.figure(1)
    plt.plot(ts, accel[0,:])
    plt.plot(ts, accel[1,:])
    plt.plot(ts, accel[2,:])
    plt.legend(['x', 'y', 'z'])
    plt.title('Accelerometer Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')

    plt.figure(2)
    plt.plot(ts, gyro[0,:])
    plt.plot(ts, gyro[1,:])
    plt.plot(ts, gyro[2,:])
    plt.legend(['x', 'y', 'z'])
    plt.title('Gyroscope Data')
    plt.xlabel('Time (s)')
    plt.ylabel('angular velocity (rad/s)')

    plt.figure(3)
    plt.plot(ts, np.ones(accel.shape[1]) * 9.81, 'k--')
    plt.plot(ts, np.linalg.norm(accel, axis=0), alpha=0.5)
    plt.title('Norm of Accelerometer Data')
    plt.legend(['magnitude of acceleration', '|g| (9.81 m/s^2)'])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    
    plt.figure(4)
    plt.plot(tv, roll, alpha=0.75)
    plt.plot(tv, pitch, alpha=0.75)
    plt.plot(tv, yaw, alpha=0.75)
    plt.plot(tv, np.ones_like(tv) * np.pi / 2, 'k--', alpha=0.5)
    plt.plot(tv, np.ones_like(tv) * (-np.pi) / 2, 'k--', alpha=0.5)
    plt.plot(tv, np.ones_like(tv) * np.pi, 'k--', alpha=0.5)
    plt.plot(tv, np.ones_like(tv) * (-np.pi), 'k--', alpha=0.5)
    plt.title('RPY of Vicon')
    plt.legend(['Roll', 'Pitch', 'Yaw'])
    plt.xlabel('Time (s)')
    plt.ylabel('Radians')
    plt.show()

def plotRPY_vs_vicon(r, p, y, t, r_vicon, p_vicon, y_vicon, tv, cal_time, split = True):
    if split:
        fig, ax = plt.subplots(2, 1, figsize=(18, 12))

        ax[0].vlines(np.array([0, cal_time]), -np.pi, np.pi, linestyles='dashed', label='Calibration Window')
        ax[0].plot(t, r, alpha=0.75)
        ax[0].plot(t, p, alpha=0.75)
        ax[0].plot(t, y, alpha=0.75)
        ax[0].plot(tv, np.ones_like(tv) * np.pi / 2, 'k--', alpha=0.25)
        ax[0].plot(tv, np.ones_like(tv) * (-np.pi) / 2, 'k--', alpha=0.25)
        ax[0].plot(tv, np.ones_like(tv) * np.pi, 'k--', alpha=0.25)
        ax[0].plot(tv, np.ones_like(tv) * (-np.pi), 'k--', alpha=0.25)

        ax[0].set_title('RPY of IMU')
        ax[0].legend(['Roll', 'Pitch', 'Yaw'])
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Radians')
        ax[0].set_ylim([-1.25*np.pi, 1.25*np.pi])


        ax[1].plot(tv, r_vicon, alpha=0.75)
        ax[1].plot(tv, p_vicon, alpha=0.75)
        ax[1].plot(tv, y_vicon, alpha=0.75)
        ax[1].plot(tv, np.ones_like(tv) * np.pi / 2, 'k--', alpha=0.25)
        ax[1].plot(tv, np.ones_like(tv) * (-np.pi) / 2, 'k--', alpha=0.25)
        ax[1].plot(tv, np.ones_like(tv) * np.pi, 'k--', alpha=0.25)
        ax[1].plot(tv, np.ones_like(tv) * (-np.pi), 'k--', alpha=0.25)

        ax[1].set_title('RPY of Vicon')
        ax[1].legend(['Roll', 'Pitch', 'Yaw'])
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Radians')
        ax[1].set_ylim([-1.25*np.pi, 1.25*np.pi])

        plt.show()

    else:
        plt.plot(t, r, alpha=0.75)
        plt.plot(t, p, alpha=0.75)
        plt.plot(t, y, alpha=0.75)

        plt.plot(tv, r_vicon, alpha=0.75)
        plt.plot(tv, p_vicon, alpha=0.75)
        plt.plot(tv, y_vicon, alpha=0.75)

        plt.plot(tv, np.ones_like(tv) * np.pi / 2, 'k--', alpha=0.5)
        plt.plot(tv, np.ones_like(tv) * (-np.pi) / 2, 'k--', alpha=0.5)
        plt.plot(tv, np.ones_like(tv) * np.pi, 'k--', alpha=0.5)
        plt.plot(tv, np.ones_like(tv) * (-np.pi), 'k--', alpha=0.5)

        plt.title('RPY vs Vicon RPY')
        plt.legend(['Roll', 'Pitch', 'Yaw', 'Vicon Roll', 'Vicon Pitch', 'Vicon Yaw'])
        plt.xlabel('Time (s)')
        plt.ylabel('Radians')
        plt.show()

def calibrationPrint(accel, gyro, str=''):
    print("Average of first 500 timesteps:")
    print(f'x accel avg {str}: {np.mean(accel[0,0:500])}')
    print(f'y accel avg {str}: {np.mean(accel[1,0:500])}')
    print(f'z accel avg {str}: {np.mean(accel[2,0:500])}')
    print(f'z gyro avg {str}: {np.mean(gyro[0,0:500])}')
    print(f'x gyro avg {str}: {np.mean(gyro[1,0:500])}')
    print(f'y gyro avg {str}: {np.mean(gyro[2,0:500])}\n')

''' CAUTION MOVING FORWARD:
(1) The orientation of the IMU need not be the same as the orientation of the Vicon
coordinate frame. Plot all quantities in the arrays accel, gyro and vicon rotation
matrices to make sure you get this right. Do not proceed to implementing the filter if
you are not convinced your solution for this part is correct.

(2) The acceleration ax and ay is flipped in sign due to device design. A positive
acceleration in body-frame will result in a negative number reported by the IMU.
See the IMU manual for more insight.
'''

class State:
    def __init__(self, state_vec, state_cov):
        self.quat = Quaternion(np.float64(state_vec[0][0]), state_vec[1:4].ravel())
        self.w = state_vec[4:,0]
        self.quat_state_vec = self.with_quat()
        self.cov = state_cov
        # self.axis_angle_state_vec = self.with_axis_angle()

    def with_quat(self):
        # gives (7,1) state vector with first four elements as quaternion elements
        return np.hstack((self.quat.q, self.w)).reshape(7, 1)

    def with_axis_angle(self):
        # replaces 4-element quaternion portion of state with its axis-angle represntation.
        # This reduces the state dimensions from (7,1) to (6,1)
        return np.hstack((self.quat.axis_angle(), self.w)).reshape(self.quat.q.size+self.w.size, 1)
        

def generate_sigma_points(mean, cov):
    # Mean is shape = (7,1), and cov is (6,6) 
    # n is no. of covariance columns
    n = cov.shape[1]

    # offset is (n,2n) array where rows of sqrt(n*cov) become 
    # columns added to mean to create sigma points
    offset = np.real((np.sqrt(n) * linalg.sqrtm(cov)).T)
    offset = np.hstack((offset, -offset))

    # initialize sigma points of shape (7,2n)
    # and merely add the angular velocity (last 3) copmponents to the offset
    sig_pts = np.zeros((mean.shape[0], 2*n))
    sig_pts[-3:, :] = mean[-3:] + offset[-3:, :]

    # must convert first 3 elements of offset term to 4-element quaternion
    # then "add" them via quaternion multiplication
    for i in range(sig_pts.shape[1]):
        offset_quat = Quaternion()
        offset_quat.from_axis_angle(offset[0:3, i])
        mean_quat = Quaternion(np.float64(mean[0]), mean[1:4].ravel())
        combo_quat  = offset_quat * mean_quat
        sig_pts[0:4, i] = combo_quat.q

    return sig_pts


def compute_GD_update(sig_pts, prev_state, threshold = 0.001):
    # TODO: rewrite/check function to match study group notes

    # Initialize mean quat to previous state's quaternion
    q_bar = Quaternion(np.float64(prev_state.quat.q[0]), prev_state.quat.q[1:4].ravel())
    # Initialize error matrix (contains axis-angle representation of error)
    E = np.ones((3, sig_pts.shape[1])) * np.inf
    mean_err = np.inf

    # Iterate until error is below threshold
    max_counts = 150
    count = 0
    while mean_err > threshold and max_counts > count:
        for i in range(sig_pts.shape[1]):
            # Convert sigma point to quaternion
            q_i = Quaternion(np.float64(sig_pts[0, i]), sig_pts[1:4, i].ravel())
            # Compute error quaternion
            q_err = q_i.__mul__(q_bar.inv())
            # Convert error quaternion to axis-angle representation
            q_err.normalize()
            E[:, i] = q_err.axis_angle()

        e_bar = np.mean(E, axis=1)
        e_bar_quat = Quaternion()
        e_bar_quat.from_axis_angle(e_bar)

        q_bar = e_bar_quat * q_bar
        mean_err = np.linalg.norm(e_bar)
        count += 1

    new_mean = np.zeros((7,1))
    new_mean[0:4] = q_bar.q.reshape(4,1)
    new_mean[4:] = np.mean(sig_pts[4:, :], axis=1).reshape(3,1)

    e_bar = e_bar.reshape(3,1)

    new_cov = np.zeros((6,6))
    #can use np.cov
    # new_cov[:3, :3] = (E - e_bar) @ (E - e_bar).T / sig_pts.shape[1] #(3, 2n) @ (2n, 3) = (3, 3)
    new_cov[:3, :3] = E @ E.T / sig_pts.shape[1] #(3, 2n) @ (2n, 3) = (3, 3)
    new_cov[3:, 3:] =(sig_pts[4:, :] - new_mean[4:]) @ (sig_pts[4:, :] - new_mean[4:]).T / sig_pts.shape[1] # + R

    return new_mean, new_cov


def propagate_dynamics(sp, dt, R, use_noise=False):
    
    sp_propagated = np.zeros(sp.shape)

    if use_noise:
        # add noise to each sigma point along with dynamics
        # will likely not do this, as we added noise to covariance before
        # computing sigma points, so we are already accounting for noise
        rng = np.random.default_rng(1998)
        noise = np.zeros((sp.shape[0]-1, 1))       

        for i in range(sp.shape[1]):
            
            for n_idx in range(sp.shape[0]-1):
                noise[n_idx] = rng.normal(0, R[n_idx,n_idx], ())

            q_delta = Quaternion()
            q_delta.from_axis_angle(sp[-3:, i] * dt)

            q_noise = Quaternion()
            q_noise.from_axis_angle(noise[0:3, 0])

            q_sp = Quaternion(np.float64(sp[0, i]), sp[1:4, i].ravel())

            q_comb = q_sp * q_noise * q_delta

            sp_propagated[0:4,i] = q_comb.q
            sp_propagated[4:,i] = sp[4:,i] + noise[3:,0]

        return sp_propagated

    else:
        # just propagate sigma points through dynamics
        for i in range(sp.shape[1]):

            q_delta = Quaternion()
            q_delta.from_axis_angle(sp[-3:, i] * dt)

            q_sp = Quaternion(np.float64(sp[0, i]), sp[1:4, i].ravel())

            q_comb = q_sp * q_delta

            sp_propagated[0:4,i] = q_comb.q
            sp_propagated[4:,i] = sp[4:,i]

        return sp_propagated
    
def propagate_measurement(sp, g_w, use_noise=False):
    sp_propagated = np.zeros((sp.shape[0]-1, sp.shape[1]))
    if use_noise:
        print(f'\nWARNING: Noise not implemented for measurement propagation\n')
        return 0
    else:
        for i in range(sp.shape[1]):
            q_sp = Quaternion(np.float64(sp[0, i]), sp[1:4, i].ravel())
            q_g = Quaternion(0, g_w.ravel())

            #EK paper wrong??? inv first???
            # q_comb = q_sp * q_g * q_sp.inv()
            q_comb = q_sp.inv() * q_g * q_sp
            g_prime = q_comb.vec()

            sp_propagated[0:3,i] = g_prime
            sp_propagated[3:,i] = sp[-3:,i]

        return sp_propagated
    
def compute_MM_mean_cov(sp_p, sp_m, mean_k1_k, Q):
    # sp_p = dynamics propagated sigma points
    # sp_m = measurement propagated sigma points
    
    # doesn't account for weights
    y_bar = np.mean(sp_m, axis=1).reshape(sp_m.shape[0], 1)

    sp_minus_ybar = sp_m - y_bar
    sp_minus_ybar_T = sp_minus_ybar.T

    Sigma_yy = Q + sp_minus_ybar @ sp_minus_ybar_T / sp_m.shape[1]

    quat_mean = Quaternion(np.float64(mean_k1_k[0]), mean_k1_k[1:4].ravel())

    sp_minus_mean = np.zeros_like(sp_m)
    for i in range(sp_m.shape[1]):
        quat_sp_p = Quaternion(np.float64(sp_p[0, i]), sp_p[1:4, i].ravel())
        # quat_sp.from_axis_angle(sp[0:3, i])
        quat_err = quat_sp_p * quat_mean.inv()
        #scalar element going negative...
        quat_err.normalize()
        aa_err = quat_err.axis_angle()
        sp_minus_mean[0:3, i] = aa_err
        sp_minus_mean[3:, i] = sp_p[-3:, i] - mean_k1_k[-3:, 0]

    Sigma_xy = sp_minus_mean @ sp_minus_ybar_T / sp_p.shape[1]

    return y_bar, Sigma_yy, Sigma_xy


def estimate_rot(data_num=1):
    # TODO: factor in WARNINGS from handout (negative axes, etc.)
    # TODO: make weights for sigma points (instead of just doing np.means)

    # accel, gyro, T_sensor, roll_gt, pitch_gt, yaw_gt, T_vicon, cal_time = preprocess_dataset(data_num)
    accel, gyro, T_sensor = preprocess_dataset_no_vicon(data_num)
    # plotStuff(accel, gyro, T_sensor.ravel(), roll_gt, pitch_gt, yaw_gt, T_vicon.ravel())    
    
    ### (1) Initialize Parameters
    # init covariance matrix
    cov_0 = np.eye(6, 6) * 0.1
    # init state
    state_0 = State(np.array([1, 0, 0, 0, 0, 0, 0]).reshape(7,1), cov_0)
    state_0.quat.from_axis_angle(-np.pi*np.array([0, 0, 1]))
    # init process noise covariance matrix
    R = np.diag([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    # init measurement noise covariance matrix
    Q = np.diag([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    # init gravity vector
    g_w = np.array([0, 0, -9.81])

    means = [state_0]
    mean_k_k = state_0.quat_state_vec
    cov_k_k = cov_0

    for t in range(T_sensor.size - 1):
        ### (2) Add Noise Component to Covariance
        dt = T_sensor[t+1] - T_sensor[t]
        cov_k_k += R * dt

        ### (3) Generate Sigma Points
        sp = generate_sigma_points(mean_k_k, cov_k_k)

        ### (4) Propagate Sigma Points Thru Dynamics
        sp_propagated = propagate_dynamics(sp, dt, R, use_noise=False)

        ### (5) Compute Mean and Covariance of Propagated Sigma Points
        mean_k1_k, cov_k1_k = compute_GD_update(sp_propagated, means[t])

        ### (6) Compute Sigma Points with Updated Mean and Covariance
        sp_measurement = generate_sigma_points(mean_k1_k, cov_k1_k)

        ### (7) Propagate Sigma Points Thru Measurement Model
        # Note: rotation component of these new sp's are in axis-angle space (6,2n) NOT quaternion space (7,2n)
        sp_measurement_propagated = propagate_measurement(sp_measurement, g_w, use_noise=False)

        ### (8) Compute Mean and Covariance of Measurement-Model Propagated Sigma Points
        sp_propagated_mean, Sigma_yy, Sigma_xy = compute_MM_mean_cov(sp_propagated, sp_measurement_propagated, mean_k1_k, Q)

        ### (9) Compute Kalman Gain
        K = Sigma_xy @ np.linalg.inv(Sigma_yy)

        ### (10) Compute Innovation, Update Mean and Covariance
        K_x_innovation = K @ (np.vstack((accel[:, t+1].reshape(3,1), gyro[:, t+1].reshape(3,1))) - sp_propagated_mean)
        quat_Ki = Quaternion()
        quat_Ki.from_axis_angle(K_x_innovation[0:3, 0])
        K_x_innovation_quat = np.vstack((quat_Ki.q.reshape(4,1), K_x_innovation[3:]))

        mean_k1_k1 = mean_k1_k +  K_x_innovation_quat
        cov_k1_k1 = cov_k1_k - K @ Sigma_yy @ K.T

        means.append(State(mean_k1_k1, cov_k1_k1))

        cov_k_k = cov_k1_k1
        mean_k_k = mean_k1_k1
        if t > 0 and t % 500 == 0:
            print(f'End of Loop {t}')

    print(f'No. of Timesteps Processed: {len(means)}')

    roll, pitch, yaw = np.zeros(T_sensor.size), np.zeros(T_sensor.size), np.zeros(T_sensor.size)
    for i in range(len(means)):
        means[i].quat.normalize()
        state_i_eas = means[i].quat.euler_angles()
        roll[i] = state_i_eas[0]
        pitch[i] = state_i_eas[1]
        yaw[i] = state_i_eas[2]

    
    # plotRPY_vs_vicon(roll, pitch, yaw, T_sensor, roll_gt, pitch_gt, yaw_gt, T_vicon, cal_time)

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw


#### TESTING ####
if __name__ == '__main__':
    _ = estimate_rot(1)
    roll, pitch, yaw = estimate_rot(3)
    print(f'DONE')