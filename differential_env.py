
from data_loader import load_test_dataset_no_cae
import matplotlib.patches as patches
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

from robot_env import RobotEnv


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

        self.dwa = DWA(self)

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
    
    def set_dwa(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'v_res': self.dwa.v_res = value
            elif key == 'w_res': self.dwa.w_res = value
            elif key == 'dt': self.dwa.dt = value
            elif key == 'predict_time': self.dwa.predict_time = value
            elif key == 'to_goal_cost_gain': self.dwa.to_goal_cost_gain = value 
            elif key == 'ob_gain': self.dwa.ob_gain = value
            elif key == 'speed_cost_gain': self.dwa.speed_cost_gain = value 
            elif key == 'skip': self.dwa.skip = value 
    
    @staticmethod
    def plot_robot(ax, x, y, yaw, robot_size):
        circle = patches.Circle(np.array([x, y]), robot_size)
        ax.add_patch(circle)

    @staticmethod
    def plot_arrow(x, y, yaw, length=0.5, width=0.1): 
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                head_length=width, head_width=width, fc='k', ec='k', zorder=0)

    @staticmethod
    def plot_ob(ax, obs_list, obs_size):
        for obs in obs_list:
            circle = patches.Circle(obs, obs_size)
            # Add the patch to the Axes
            ax.add_patch(circle)



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
        self.skip=3

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
    



def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def plot_ob(ax, obs_list, obs_size):
    for obs in obs_list:
        circle = patches.Circle(obs, obs_size)
        # Add the patch to the Axes
        ax.add_patch(circle)

if __name__ == "__main__":
    obc = load_test_dataset_no_cae()

    env = DifferentialDriveEnv(1.0, -0.1, math.pi, 1.0, math.pi)
    
    env.set_obs(obc[0])
    env.set_dwa(dt=0.2, to_goal_cost_gain=1.2)
    dwa = env.dwa
    
    goal = np.array([1.66310299e+01, 4.11202216e+00,0,0,0.0])
    start = np.array([1.54019751e+01, 1.33907094e+01,0,0,0.0])
    state = start.copy()
    fig, ax = plt.subplots()
    for i in range(200):
        v, traj = dwa.control(state, goal)

        print(v)
        state = env.motion_velocity(state, v, 1.0/5.0)
        
        if not env.valid_state_check(state): 
            print('Collision')
            break
        if np.linalg.norm(state[:2]-goal[:2])<.6: 
            print('Goal')
            break
        plt.cla()
        plot_ob(ax, env.obs_list, env.obs_size)
        plt.plot(traj[:, 0], traj[:, 1], "-c")
        plt.plot(state[0], state[1], "xr")
        plt.plot(goal[0], goal[1], "xb")
        plot_arrow(state[0], state[1], state[2])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
    plt.show()
        