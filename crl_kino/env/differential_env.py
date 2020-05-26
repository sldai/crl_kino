


from itertools import product
from math import cos, sin, atan2, pi
import math
#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt

from crl_kino.env.robot_env import RobotEnv



def normalize_angle(angle):
    norm_angle = angle % (2 * math.pi)
    if norm_angle > math.pi:
        norm_angle -= 2 * math.pi
    return norm_angle


class DifferentialDriveEnv(RobotEnv):
    def __init__(self, max_v, min_v, max_w, max_acc_v, max_acc_w,
                 robot_radius=0.3, obs_size=2.5, env_bounds=[-20, 20, -20, 20], base_dt=0.1):
        super(DifferentialDriveEnv, self).__init__()

        # physical constrain
        self.max_v = max_v
        self.min_v = min_v
        self.max_w = max_w
        self.min_w = -self.max_w
        self.max_acc_v = max_acc_v
        self.min_acc_v = -self.max_acc_v
        self.max_acc_w = max_acc_w
        self.min_acc_w = -self.max_acc_w
        self.robot_radius = robot_radius

        self.env_bounds = np.zeros((2, 2))
        self.env_bounds[0, 0] = env_bounds[0]
        self.env_bounds[0, 1] = env_bounds[1]
        self.env_bounds[1, 0] = env_bounds[2]
        self.env_bounds[1, 1] = env_bounds[3]

        # obstacles
        self.obs_list = np.empty((0, 2))
        self.obs_size = obs_size

        self.base_dt = base_dt


    def motion_velocity(self, state, velocity, dt):
        input_u = np.zeros(2)
        input_u[0] = (velocity[0] - state[3])/dt
        input_u[1] = (velocity[1] - state[4])/dt
        return self.motion(state, input_u, dt)

    def motion(self, state, input_u, duration):
        t = 0
        state = state.copy()

        while t < duration:
            state = self.motion_base(state, input_u, self.base_dt)
            t += self.base_dt
        return state

    def motion_base(self, state, input_u, dt):
        # constrain input velocity
        constrain = self.dynamic_window(state, dt)
        u = np.zeros(2)
        u[0] = state[3] + input_u[0]*dt
        u[1] = state[4] + input_u[1]*dt
        u[0] = max(constrain[0,0], u[0])
        u[0] = min(constrain[0,1], u[0])

        u[1] = max(constrain[1,0], u[1])
        u[1] = min(constrain[1,1], u[1])
        # motion model
        state[2] += u[1] * dt

        state[2] = normalize_angle(state[2])
        state[0] += u[0] * cos(state[2]) * dt
        state[1] += u[0] * sin(state[2]) * dt
        state[3] = u[0]
        state[4] = u[1]

        return state

    def dynamic_window(self, state, dt):
        # dynamic constrain
        v_pre_max = min(state[3] + self.max_acc_v * dt, self.max_v)
        v_pre_min = max(state[3] - self.max_acc_v * dt, self.min_v)
        w_pre_max = min(state[4] + self.max_acc_w * dt, self.max_w)
        w_pre_min = max(state[4] - self.max_acc_w * dt, -self.max_w)

        return np.array([[v_pre_min, v_pre_max], [w_pre_min, w_pre_max]])

    def add_obs(self, obs_list: np.ndarray):
        self.obs_list = np.vstack((self.obs_list, obs_list))
    
    def set_obs(self, obs_list: np.ndarray):
        self.obs_list = obs_list.copy()

    def set_obs_size(self, obs_size): self.obs_size = obs_size

    def get_clearance(self, state):
        dis = 100
        if len(self.obs_list) > 0:
            dis = np.linalg.norm(self.obs_list - state[:2], axis=1)
            dis = np.min(dis) - self.obs_size - self.robot_radius
        return dis

    def valid_point_check(self, point):
        state = np.zeros(5)
        state[:2] = point[:2]
        dis = self.get_clearance(state) + self.robot_radius
        return dis>0

    def valid_state_check(self, state):
        dis = self.get_clearance(state)
        return dis >= 0

    def get_bounds(self):
        state_bounds = np.zeros((5, 2))
        state_bounds[:2, :] = self.env_bounds
        state_bounds[2, 0] = -np.pi
        state_bounds[2, 1] = np.pi
        state_bounds[3, 0] = self.min_v
        state_bounds[3, 1] = self.max_v
        state_bounds[4, 0] = self.min_w
        state_bounds[4, 1] = self.max_w

        control_bounds = np.zeros((2, 2))
        control_bounds[0, 0] = self.min_acc_v
        control_bounds[0, 1] = self.max_acc_v
        control_bounds[1, 0] = self.min_acc_w
        control_bounds[1, 1] = self.max_acc_w
        bounds = {'state_bounds': state_bounds, 'control_bounds': control_bounds}
        return bounds
    

    








        