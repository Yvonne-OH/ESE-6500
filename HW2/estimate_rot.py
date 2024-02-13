import numpy as np
from scipy import io
from quaternion import Quaternion
import math

import matplotlib.pyplot as plt



#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    # your code goes here

    # roll, pitch, yaw are numpy arrays of length T
    return 0
    #return roll,pitch,yaw

data_num=1

imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
accel = imu['vals'][0:3,:]
gyro = imu['vals'][3:6,:]
T = np.shape(imu['ts'])[1]
dt = imu['ts']

acc_sensitivity = 33.6
V_ref=3300
g=9.81

acc_x = np.array(accel[0])
acc_y = np.array(accel[1])
acc_z = np.array(accel[2])
acc = np.array([acc_x, acc_y, acc_z]).T

acc_scale_factor = V_ref/(1023.0*acc_sensitivity)/g
acc_bias = np.mean(acc[:10], axis=0) - np.array([0,0,1])/acc_scale_factor
acc = (acc-acc_bias)*acc_scale_factor

acc_x = np.array(acc[:,0]) # ax and ay is flipped
acc_y = -np.array(acc[:,1])
acc_z = np.array(acc[:,2])

Pitch = np.arctan(acc_x / np.sqrt(acc_y**2 + acc_z**2))
Roll =  np.arctan(acc_y / np.sqrt(acc_x**2 + acc_z**2))
Yaw =   np.arctan2(acc_y, acc_x)

rots_vicon = vicon['rots']
T_vicon = (vicon['ts'])


Roll_vicon = np.arctan2(rots_vicon[2,1,:], rots_vicon[2,2,:])
Pitch_vicon = np.arcsin(-rots_vicon[2,0,:])
Yaw_vicon = np.arctan2(rots_vicon[1,0,:], rots_vicon[0,0,:])


#plt.plot(acc_x)
#plt.plot(acc_y)
plt.plot(Pitch, label='Pitch') 
plt.plot(Pitch_vicon, label='Pitch Vicon')  
plt.legend()  
plt.grid(True)  
plt.show()

plt.plot(Roll)
plt.plot(Roll_vicon)
plt.grid()
plt.show()

"""
plt.plot(Yaw)
plt.plot(Yaw_vicon)
plt.grid()
plt.show()
"""



gyro_x = np.array(gyro[0]) # angular rates are out of order !
gyro_y = np.array(gyro[1])
gyro_z = np.array(gyro[2])
gyro = np.array([gyro_x, gyro_y, gyro_z]).T




gyro_sensitivity = 3.33
gyro_bias = np.mean(gyro[:20], axis=0)
gyro_scale_factor = V_ref/1023/gyro_sensitivity
gyro = (gyro-gyro_bias)*gyro_scale_factor*(np.pi/180)



orientation = np.zeros((gyro.shape[0],3))  # [pitch, roll, yaw]
orientation[0,:] = [Pitch[0], Roll[0], 0]  # 偏航角初始值假设为0

# 通过陀螺仪数据积分更新方向
for i in range(1, len(gyro)-1):
   
    dp = gyro[i-1,0] * (dt[0, i]-dt[0, i-1])
    dr = gyro[i-1,1] * (dt[0, i]-dt[0, i-1])
    dy = gyro[i-1,2] * (dt[0, i]-dt[0, i-1])
    
    orientation[i,0] = orientation[i-1,0] + dp
    orientation[i,1] = orientation[i-1,1] + dr
    orientation[i,2] = orientation[i-1,2] + dy

plt.plot( orientation[:,0])
plt.plot( orientation[:,1])
plt.plot( orientation[:,2])
plt.plot(Pitch_vicon, label='Pitch Vicon')
plt.plot(Roll_vicon, label='Roll Vicon')
plt.legend()  
plt.grid()
plt.show()
