#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt

from crl_kino.env.differential_env import DifferentialDriveEnv

class DWA():
    def __init__(self, robot_env: DifferentialDriveEnv, v_res = 0.01, w_res = 0.1, dt=0.1, predict_time=1.0, to_goal_cost_gain=1.0, ob_gain=1.0, speed_cost_gain=0.3, skip = 3):
        self.robot_env = robot_env
        self.v_res = v_res
        self.w_res = w_res
        self.dt = dt  # [s]
        self.predict_time = predict_time  # [s]
        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_cost_gain = speed_cost_gain
        self.ob_gain = ob_gain
        self.skip=skip

    def set_dwa(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'v_res': self.v_res = value
            elif key == 'w_res': self.w_res = value
            elif key == 'dt': self.dt = value
            elif key == 'predict_time': self.predict_time = value
            elif key == 'to_goal_cost_gain': self.to_goal_cost_gain = value 
            elif key == 'ob_gain': self.ob_gain = value
            elif key == 'speed_cost_gain': self.speed_cost_gain = value 
            elif key == 'skip': self.skip = value 

    def calc_trajectory(self, x_init, v):
        x = np.array(x_init)
        traj = x.copy()
        t = 0
        while t <= self.predict_time:
            x = self.robot_env.motion_velocity(x, v, self.dt)
            traj = np.vstack((traj, x))
            t += self.dt
        return traj

    def control(self, x_init, goal):
        dw = self.robot_env.dynamic_window(x_init, self.dt)

        v_list = []
        cost_list = np.empty((0, 3))
        traj_list = []
        end_point = 1
        for v in np.arange(dw[0,0], dw[0,1]+self.v_res, self.v_res):
            for w in np.arange(dw[1,0], dw[1,1]+self.w_res, self.w_res):
                v_list.append(np.array([v,w]))
                traj = self.calc_trajectory(x_init, np.array([v,w]))
                traj_list.append(traj)
                # calc to goal cost. It is 2D norm.
                goal_line_angle = np.arctan2(
                    goal[1] - traj[end_point, 1], goal[0]-traj[end_point, 0])
                
                to_goal_cost = abs(goal_line_angle-traj[end_point, 2])
                to_goal_cost = min(to_goal_cost, 2*np.pi-to_goal_cost)
                # calc clearance cost
                min_dis = 100
                for i in range(0,len(traj),self.skip):
                    x = traj[i]
                    dis = max(self.robot_env.get_clearance(x), 0.001)
                    min_dis = min(dis, min_dis)
                min_dis = 1.0/min_dis

                # calc speed cost
                speed_cost = dw[0, 1]-traj[-1, 3]
                new_cost_entry = np.array([to_goal_cost, min_dis, speed_cost])
                cost_list = np.vstack((cost_list, new_cost_entry))
        v_list = np.array(v_list)
        for i in range(len(cost_list[0])):
            if np.sum(cost_list[:, i]) != 0:
                cost_list[:, i] /= np.sum(cost_list[:, i])
        cost_sum = self.to_goal_cost_gain * \
            cost_list[:, 0] + self.ob_gain*cost_list[:, 1] + \
            self.speed_cost_gain * cost_list[:, 2]
        opt_ind = np.argmin(cost_sum)
        opt_v = np.array(v_list[opt_ind])
        opt_traj = traj_list[opt_ind]

        return opt_v, opt_traj   

    