#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
from operations import *

def estimate_rot(data_num=1):
    imu1 = loadmat('imu/imuRaw' + str(data_num) + '.mat')
    
    Vref = 3300
    Vref = float(Vref)
    s_acc = 339
    scale_factor_acc = Vref/1023/s_acc
    bias_acc = imu1['vals'][:3,0] - np.array([0,0,1/scale_factor_acc])
    
    s_gyro=3.5
    scale_factor_gyro = math.pi/180*Vref/1023/s_gyro
    bias_gyro = imu1['vals'][3:6,0]
    
    '''converted imu data '''
    acc = (imu1['vals'][:3,:].T-bias_acc)*scale_factor_acc * [-1,-1,1]
    acc = acc.T
    gyro_o = ((imu1['vals'][3:6,:].astype(int).T-bias_gyro)*scale_factor_gyro).T
    gyro = np.zeros(np.shape(gyro_o))
    gyro[0] = gyro_o[1]
    gyro[1] = gyro_o[2]
    gyro[2] = gyro_o[0]
    
    P = 1 * np.identity(3)
    Q = 0.001*np.identity(3)
    R = 0.1 * np.identity(3)
    n = P.shape[0]
    qk = np.array([1,0,0,0])
    g = np.array([0,0,0,1])
    ts = imu1['ts'][0]
    
    l1 = np.shape(imu1['vals'])[1]-1
    
    roll = np.zeros(l1)
    pitch = np.zeros(l1)
    yaw = np.zeros(l1)
    
    for k in range(l1):
    #for k in range(1):
        '''acc as the measurement'''
        val = acc[:,k]
        t = ts[k+1]-ts[k]
        '''gyro as the control inputs'''
        wk = gyro[:,k]   
        
        P = P + 0.004*np.identity(3)
        S = np.linalg.cholesky(P + Q)
        
        W = np.concatenate((np.sqrt(n)*S,-np.sqrt(n)*S),axis=1)
        #sigma points
        Xq = np.zeros([4,2*n])
        #Xw = np.zeros([3,12])
        '''gyro as the control inputs'''
        q_delta = vector_to_quaternion(wk*t)
        for i in range(2*n):   
            qw = vector_to_quaternion(W[:,i])        
            Xq[:,i] = quaternion_multiplication(qk,qw)          
            Xq[:,i] = quaternion_multiplication(Xq[:,i],q_delta)
     
        '''X mean, covariance'''
        e_q = np.zeros([4,2*n])
        e_v = np.zeros([3,2*n])
        for i in range(2*n):
            e_q[:,i] = quaternion_multiplication(inverse_quaternion(qk),Xq[:,i])
            e_v[:,i] = quaternion_to_vector(e_q[:,i])
        err = np.sum(e_v,axis=1)/(2*n)
        err_quaternion = vector_to_quaternion(err)
        mean_qk = quaternion_multiplication(err_quaternion,qk)
        #mean_wk = np.sum(Xw,axis=1)/12
        
        V = e_v
        Pk = np.zeros([n,n])
        for i in range(2*n):
            Pk += np.dot(V[:,i].reshape([n,1]),V[:,i].reshape([1,n]))/(2*n)
       
        '''
        measurement model
        Zi mean and covariance
        '''
        zacc = np.zeros([3,2*n])
        for i in range(2*n):
            temp = quaternion_multiplication(Xq[:,i],g)
            g_prime = quaternion_multiplication(temp,inverse_quaternion(Xq[:,i]))
            zacc[:,i] = quaternion_to_vector(g_prime)
        zacc_mean = np.sum(zacc,axis=1)/(2*n)
        #zw_mean = np.sum(zw,axis=1)/12
        '''
        Pzz,Pvv,Pxz    
        '''  
        Z = zacc - zacc_mean[:,None]
        Pzz = np.zeros([n,n])   
        for i in range(2*n):
            Pzz += np.dot(Z[:,i].reshape([n,1]),Z[:,i].reshape([1,n]))/(2*n)   
        Pvv = Pzz + R  
        Pxz = np.zeros([n,n])   
        for i in range(2*n):
            Pxz += np.dot(V[:,i].reshape([n,1]),Z[:,i].reshape([1,n]))/(2*n)
       
        '''
        Kalman gain and update
        UKF outputs:
            qk,wk,P
            
        '''
        K = np.dot(Pxz,np.linalg.inv(Pvv))
        Kv = np.dot(K,val - zacc_mean)
        qk = quaternion_multiplication(mean_qk,vector_to_quaternion(Kv))  
        #wk = mean_wk + Kv[3:]    
        P = Pk - np.dot(np.dot(K,Pvv),np.transpose(K))
        
    #    results[:,k] = quaternion_to_vector(qk)
    #    results[:,:,k] = quaternion_to_rot(qk)
    #    roll1[k],pitch1[k],yaw1[k] = rot_to_euler_angles(results[:,:,k])
    #    a = results[:,k]
    #    roll[k] = math.atan2(-a[1],a[2])
    #    pitch[k] = math.atan2(a[0],np.sqrt(a[1]**2 + a[2]**2))
    #    if k == 0:
    #        yaw[k] = 0
    #    else:
    #        yaw[k] = yaw[k-1] + wk[0] * t
        
        roll[k],pitch[k],yaw[k] = quaternion_to_euler(qk)

    return roll,pitch,yaw


