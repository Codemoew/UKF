#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 01:07:14 2019

@author: yang
"""
import math
import numpy as np

def quaternion_multiplication(q1,q2):
    q = np.zeros(4)
    q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q[1] = q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2]
    q[2] = q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0] - q1[3]*q2[1]
    q[3] = q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] + q1[3]*q2[0]
    return q

def vector_to_quaternion(w):
    q = np.zeros(4)
    q[0] = math.cos(np.linalg.norm(w)/2)
    if np.linalg.norm(w)==0:
        q[1:4] = 0
    else:
        q[1:4] = w/np.linalg.norm(w) * math.sin(np.linalg.norm(w)/2)
    return q

def norm_quaternion(q):
    norm = np.sqrt(np.sum(np.power(q,2)))
    return norm

def inverse_quaternion(q):
    q_inv = np.zeros(4)
    #n = np.sum(np.power(q,2))
    q_inv[0] = q[0]
    q_inv[1] = -q[1]
    q_inv[2] = -q[2]
    q_inv[3] = -q[3]
    return q_inv

def quaternion_to_vector(q):
    r = q[0]
    v = q[1:4]
    if np.linalg.norm(v) == 0:
        vector = np.zeros(3)
    else:
        vector = 2 * math.acos(r) * (v/np.linalg.norm(v))
    return vector


def rot_to_euler_angles(R): 
    roll = math.atan2(R[2,1],R[2,2])
    pitch = math.atan2(-R[2,0],math.sqrt(R[2,1]**2+R[2,2]**2))
    yaw =  math.atan2(R[1,0],R[0,0])
    return roll, pitch, yaw
    
def quaternion_to_rot(q):
    q =q/np.sqrt(np.sum(np.power(q,2)))
    q_hat = np.zeros([3,3])
    q_hat[0,1] = -q[3]
    q_hat[0,2] = q[2]
    q_hat[1,2] = -q[1]
    q_hat[1,0] = q[3]
    q_hat[2,0] = -q[2]
    q_hat[2,1] = q[1]
    R = np.identity(3) + 2*(np.dot(q_hat,q_hat) + q_hat*np.array(q[0]))
    return R

def quaternion_to_euler(q):
    roll = math.atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]**2+q[2]**2))
    pitch = math.asin(2*(q[0]*q[2]-q[3]*q[1]))
    yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    return roll,pitch,yaw
#
#def quaternion_to_euler(q):
#    n = np.linalg.norm(q[1:4])
#    c = q[0]
#    a = math.atan2(n,c)
#    if n==0:
#        roll=0
#        pitch=0
#        yaw=0
#    else:
#        e=q[1:4]/np.abs(n)
#        rv = e*2*np.abs(a)
#        roll = rv[0]
#        pitch = rv[1]
#        yaw = rv[2]
#    return roll,pitch,yaw
#    

