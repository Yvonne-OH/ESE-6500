import numpy as np
from scipy import io
from scipy.linalg import sqrtm
from quaternion import Quaternion
import math

import matplotlib.pyplot as plt



def gaussian_update(q, P, Q):  # checked

    n,c = P.shape                                       # (3,3)
    S = np.linalg.cholesky(P+Q)                         # SS^T=P+Q
    postive_offset = S * np.sqrt(2*n)
    negative_offset = -S * np.sqrt(2*n)
    vec = np.hstack((postive_offset, negative_offset))  # (3,6)
    X = np.zeros((2*n, 4))                              # (6,4)

    for i in range(2*n): # 6
        qW = Quaternion()                       # gyro to quaternion (4,) 
        qW.from_axis_angle(vec[:, i])           # obtain the representation in the axis-angle form                  
        X[i, :] = __mul__(q, qW.q) # add mean
    X = np.vstack((q, X)) #(7,4)                # add itself
    return X


def sigma_update(X, g, dt): # X:(7,4) g:(3,) dt:scalar
    """
     updating sigma points in quaternion-based Kalman. 
     prediction step of the filter
     
     X: A matrix of sigma points,
     g: A 3-dimensional angular velocity measurements from gyroscope
     
     Y: A matrix of updated sigma points
    """
    n = X.shape[0]                                  # 7
    Y = np.zeros((n,4))                             # (7,4)
    delta_quaternion=Quaternion()                   # compute delta quaternion
    delta_quaternion.from_axis_angle(g*dt)
    q_delta = delta_quaternion.q                    #(4,)
    for i in range(n):
        q = X[i]                                    # project sigma points by process model (4,) 
        Y[i] = __mul__(q, q_delta)                  # q_del * q_w * q
    return Y                                        #(7,4)

def compute_delta_time(i, T, imu_ts):
    """
    Computes the time interval (dt) between successive IMU measurements.

    Parameters:
    - i: The current index in the loop.
    - T: The total number of time steps.
    - imu_ts: A list or array of timestamp values for the IMU measurements.

    Returns:
    - dt: The computed time interval between the current and the next measurement.
    """
    if i == T-1:
        dt = imu_ts[-1] - imu_ts[-2]
    else:
        dt = imu_ts[i+1] - imu_ts[i]
    
    return dt

def estimate_rot(data_num=1):
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    #vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]
    
    imu_vals = imu['vals']
    imu_ts = imu['ts']
    imu_ts = np.array(imu['ts']).T  # (5000,1)

    Vref = 3300
    acc_sensitivity = 330.0



    acc_x = -np.array(imu_vals[0]) # IMU Ax and Ay direction is flipped !
    acc_y = -np.array(imu_vals[1])
    acc_z = np.array(imu_vals[2])
    acc = np.array([acc_x, acc_y, acc_z]).T

    
    acc_scale_factor = Vref/1023.0/acc_sensitivity
    acc_bias = np.mean(acc[:10], axis=0) - np.array([0,0,1])/acc_scale_factor
    acc = (acc-acc_bias)*acc_scale_factor


    gyro_x = np.array(imu_vals[4]) # angular rates are out of order !
    gyro_y = np.array(imu_vals[5])
    gyro_z = np.array(imu_vals[3])
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T
    gyro_sensitivity = 3.33
    gyro_bias = np.mean(gyro[:10], axis=0)
    gyro_scale_factor = Vref/1023/gyro_sensitivity
    gyro = (gyro-gyro_bias)*gyro_scale_factor*(np.pi/180)

    imu_vals = np.hstack((acc,gyro)) # updated imu_vals  (5000,6)

    P = np.eye(3, 3) * 0.1      # init covariance matrix
    Q = np.eye(3, 3) * 2.0      # init Process Noise Covariance Matrix
    R = np.eye(3, 3) * 2.0      # init Measurement Noise Covariance Matrix
    qt = np.array([1, 0, 0, 0]) # init initial quaternion 
    predicted_q = qt            # init predicted quaternion

    for i in range(T):

        acc = imu_vals[i,:3]  # (3, )
        gyro = imu_vals[i,3:] # (3, )
        
        dt = compute_delta_time(i, T, imu_ts)

        """
        An intermediate prediction using
        current orientation estimate, 
        process noise covariance (Q), 
        current estimate error covariance (P).
        """
        X = gaussian_update(qt, P, Q)  # (7,4)=(q,w)
        # Process model
        Y = sigma_update(X, gyro, dt) #(7,4)
        
        # compute mean
        x_k_bar, error = quat_average(Y, qt) # 38,39 # (4,)  error:(7,3)
        
        
        P_k_bar = np.zeros((3, 3))                          # (3,3)
        for i in range(7):
            P_k_bar += np.outer(error[i,:], error[i,:])     # compute covariance
        P_k_bar /= 7

        
        g = np.array([0, 0, 0, 1])                          # measurement model

        Z = np.zeros((7, 3))
        for i in range(7):                                  # compute predicted acceleration
            q = Y[i]
            q_Q=Quaternion(q[0],-q[1:])
            q_Q.inv
            Z[i] = __mul__(__mul__(q_Q.q, g), q)[1:]        # rotate from body frame to global frame
    
        
        z_k_bar = np.mean(Z, axis=0)                        # measurement mean
        z_k_bar /= np.linalg.norm(z_k_bar)
        Pzz = np.zeros((3, 3))                              # measurement cov and correlation
        Pxz = np.zeros((3, 3))
        Z_err = Z - z_k_bar
        for i in range(7):
            Pzz += np.outer(Z_err[i,:], Z_err[i,:])
            Pxz += np.outer(error[i,:], Z_err[i,:])
        Pzz /= 7
        Pxz /= 7
        
        
        acc /= np.linalg.norm(acc)                          # compute innovation
        vk = acc - z_k_bar                                  # EK 44
        Pvv = Pzz + R                                       # EK 45
        K = np.dot(Pxz,np.linalg.inv(Pvv))                  # compute Kalman gain EK 72 
        
        
        # states update
        qt,P = kalman_update(P_k_bar, K, vk, x_k_bar, Pvv)
        # predict
        predicted_q = np.vstack((predicted_q, qt))

    return quat_to_euler_angles(predicted_q)

def kalman_update(P_k_bar, K, vk, x_k_bar, Pvv):
    gain=Quaternion()
    gain.from_axis_angle(K.dot(vk))
    q_gain = gain.q                                     # EK 74
    q_update = __mul__(q_gain,x_k_bar)                  # EK 75
    P_update = P_k_bar - K.dot(Pvv).dot(K.T)
    return q_update, P_update


def quat_average(q, q0): # checked # q:(7,4) q0:(4,)
    """
    compute the average of a set of quaternions. 
    
    q:  (7,4) sigma of 7 quaternions.
    q0: (4,)  current pose estimation quaternion.
    Procedure.
    
    qt: an updated mean quaternion representing the best pose estimate.
    error: (7,3) rotation error vector between each quaternion and the qt.
    """
    qt = q0
    r, c = q.shape #r=7,
    epsilon = 0.0001
    error = np.zeros((r,3)) #(7,3)
    
    for _ in range(1000):
        for i in range(r):
            qt_Q=Quaternion(qt[0],-qt[1:])
            qt_Q.inv

            qi_error = normalize_quaternion(__mul__(q[i, :],qt_Q.q)) # (52)
            

            qs = qi_error[0]
            qv = qi_error[1:4]
            if np.linalg.norm(qv) == 0:
                ev_error = np.zeros(3)
            else:
                ev_error = 2*qv/np.linalg.norm(qv)*np.arccos(qs/np.linalg.norm(qi_error))


            if np.linalg.norm(ev_error) == 0: # not rotate
                error[i:] = np.zeros(3)
            else:
                error[i,:] = (-np.pi + np.mod(np.linalg.norm(ev_error) + np.pi, 2 * np.pi)) / np.linalg.norm(ev_error) * ev_error
        
        error_mean = np.mean(error, axis=0)
        
        em=Quaternion()
        em.from_axis_angle(error_mean)
        qt = normalize_quaternion(__mul__(em.q, qt))
        
        if np.linalg.norm(error_mean) < epsilon:
            return qt, error
        error = np.zeros((r,3))

def quat_to_euler_angles(predicted_q):
    """
    Convert a series of quaternions to Euler angles (roll, pitch, yaw).
    
    Parameters:
    - predicted_q: A numpy array of quaternions, shape (N, 4), where N is the number of quaternions.
    
    Returns:
    - roll: A numpy array of roll angles, shape (N,).
    - pitch: A numpy array of pitch angles, shape (N,).
    - yaw: A numpy array of yaw angles, shape (N,).
    """
    N = predicted_q.shape[0]
    roll = np.zeros(N)
    pitch = np.zeros(N)
    yaw = np.zeros(N)

    for i in range(N):
        roll[i] = np.arctan2(2*(predicted_q[i][0]*predicted_q[i][1]+predicted_q[i][2]*predicted_q[i][3]),
                             1 - 2*(predicted_q[i][1]**2 + predicted_q[i][2]**2))
        pitch[i] = np.arcsin(2*(predicted_q[i][0]*predicted_q[i][2] - predicted_q[i][3]*predicted_q[i][1]))
        yaw[i] = np.arctan2(2*(predicted_q[i][0]*predicted_q[i][3]+predicted_q[i][1]*predicted_q[i][2]),
                            1 - 2*(predicted_q[i][2]**2 + predicted_q[i][3]**2))

    return roll, pitch, yaw

def __mul__(a, b):
    """
    Inherited from the quaternion class.
    """

    t0 = a[0]*b[0] - \
         a[1]*b[1] - \
         a[2]*b[2] - \
         a[3]*b[3]
    t1 = a[0]*b[1] + \
         a[1]*b[0] + \
         a[2]*b[3] - \
         a[3]*b[2]
    t2 = a[0]*b[2] - \
         a[1]*b[3] + \
         a[2]*b[0] + \
         a[3]*b[1]
    t3 = a[0]*b[3] + \
         a[1]*b[2] - \
         a[2]*b[1] + \
         a[3]*b[0]
    retval = np.array([t0,t1,t2,t3])
    return retval


def normalize_quaternion(q): 
    return q/np.sqrt(np.sum(np.power(q, 2)))



if __name__ == '__main__':
     # _ = estimate_rot(1)
     roll, pitch, yaw = estimate_rot(3)
     print(f'DONE')