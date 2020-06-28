#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from crl_kino.env.dubin_env import DubinEnv, normalize_angle
import itertools
import transforms3d.euler as euler
import pickle
from crl_kino.utils.draw import *
from tianshou.data import Batch


class DubinGym(gym.Env):
    """
    The gym wrapper for DifferentailDriveEnv
    """
    def __init__(self,
                robot_env=DubinEnv(),
                 obs_list_list=[]):
        """
        :param robot_env: simulation environment
        """
        super().__init__()
        #
        self.robot_env = robot_env
        self.state_bounds = self.robot_env.get_bounds()['state_bounds'].copy()
        self.cbounds = self.robot_env.get_bounds()['control_bounds'].copy()
        self.state = np.zeros(len(self.state_bounds))

        self.goal = np.zeros(len(self.state))

        self.action_space = spaces.Box(
            low=self.cbounds[:, 0], high=self.cbounds[:, 1])

        self._init_sample_positions()

        # environment observations
        self.local_map_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.local_map_shape)
        # current and target states
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,))
        self.observation_space = spaces.Tuple(
            (self.local_map_space, self.state_space))

        self.n_step = 0
        self.max_step = 200
        self.current_time = 0.0
        self.max_time = 100.0

        self.obs_list_list = obs_list_list

    def _init_sample_positions(self, left=-5, right=5, backward=-4, forward=10):
        """
        the robot can sense the local environment by sampling the local map.

        Returns
        -------
        sample_positions contain the positions need to be sampled in the body coordinate system.
        """
        local_map_size = np.array(
            [left, right, backward, forward])  # left, right, behind, ahead
        sample_reso = 0.3
        lr = np.arange(local_map_size[0], local_map_size[1], sample_reso)
        ba = np.arange(local_map_size[2], local_map_size[3], sample_reso)
        sample_positions = np.array(list(itertools.product(ba, lr)))
        self.local_map_size = local_map_size
        # (channel, width, height)
        self.local_map_shape = (1, len(ba), len(lr))
        self.sample_reso = sample_reso
        self.sample_positions = sample_positions

    def sample_local_map(self):
        """
        sampling points of local map, with value representing occupancy (0 is non-free, 255 is free)
        wTb: world transform robot
        """
        wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        # world sample position
        wPos = self.T_transform2d(wTb, self.sample_positions)
        local_map = self.robot_env.valid_point_check(wPos).astype(np.float32)
        return local_map

    def step(self, action):
        dt = 1.0/5.0  # 5 Hz
        self.state = self.robot_env.motion(self.state, action, dt)
        self.current_time += dt

        # obs
        obs = self._obs()

        # info
        info = {'goal': False,
                'goal_dis': 0.0,
                'collision': False,
                'clearance': 0.0,
                }
        x1 = np.block([self.state[:2], np.cos(
            self.state[2]), np.sin(self.state[2])])
        x2 = np.block([self.goal[:2], np.cos(
            self.goal[2]), np.sin(self.goal[2])])
        info['goal_dis'] = np.linalg.norm(x1-x2) # SE(2) distance
        if info['goal_dis'] <= 1.5:
            info['goal'] = True
        info['clearance'] = min(self.robot_env.get_clearance(self.state), 3.0)
        info['collision'] = not self.robot_env.valid_state_check(self.state)

        # reward
        reward = -0.5*info['goal_dis']+0.2*np.tanh(action[0])-1.0*np.tanh(action[1])+20.0*info['goal']-50.0*info['collision']+0.1*info['clearance']

        # done
        done = info['goal'] or info['collision'] or self.current_time >= self.max_time

        return obs, reward, done, info

    def _obs(self):
        local_map = self.sample_local_map()
        local_map = local_map.reshape(self.local_map_shape)
        wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        bTw = np.linalg.inv(wTb)
        b_goal_pos = self.T_transform2d(bTw, self.goal[:2])
        b_theta = normalize_angle(self.goal[2]-self.state[2])
        b_goal = np.block([b_goal_pos, np.cos(b_theta), np.sin(b_theta)])

        # goal configuration in the robot coordinate frame, local map
        obs = (local_map, b_goal)
        return obs
    

    @staticmethod
    def T_transform2d(aTb, bP):
        """
        aTb: transform with rotation, translation
        bP: non-holonomic coordinates in b
        return: non-holonomic coordinates in a
        """
        if len(bP.shape) == 1:  # vector
            bP_ = np.concatenate((bP, np.ones(1)))
            aP_ = aTb @ bP_
            aP = aP_[:2]
        elif bP.shape[1] == 2:
            bP = bP.T
            bP_ = np.vstack((bP, np.ones((1, bP.shape[1]))))
            aP_ = aTb @ bP_
            aP = aP_[:2, :].T
        return aP

    def reset(self, low=5, high=12):
        # assert len(self.obs_list_list) > 0, 'No training environments'
        if len(self.obs_list_list)>0:
            ind_obs = np.random.randint(0, len(self.obs_list_list))
            self.robot_env.set_obs(self.obs_list_list[ind_obs])

        # sample a random start goal configuration
        start = np.zeros(len(self.state_bounds))
        goal = np.zeros(len(self.state_bounds))
        while True:  # random sample start and goal configuration
            # sample a valid state
            start[:] = np.random.uniform(
                self.state_bounds[:, 0], self.state_bounds[:, 1])
            if self.robot_env.get_clearance(start) <= 0.5:
                continue

            # sample a valid goal
            for _ in range(5):
                r = np.random.uniform(low, high)
                theta = np.random.uniform(-np.pi, np.pi)
                goal[0] = np.clip(start[0] + r*np.cos(theta),
                                  *self.state_bounds[0, :])
                goal[1] = np.clip(start[1] + r*np.sin(theta),
                                  *self.state_bounds[1, :])
                if self.robot_env.get_clearance(goal) > 0.5:
                    break

            if self.robot_env.get_clearance(start) > 0.5 and self.robot_env.get_clearance(goal) > 0.5 and low < np.linalg.norm(start[:2]-goal[:2]) < high:
                break
        self.state = start
        self.goal = goal

        self.current_time = 0
        obs = self._obs()
        return obs

    def render(self, mode='human', plot_localwindow=True):
        if not hasattr(self, 'ax'):
            fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.xticks([])
            plt.yticks([])

        self.ax.cla()  # clear things

        plot_obs_list(self.ax, self.robot_env.obs_list)
        if plot_localwindow:
            self.plot_observation()
        plot_problem_definition(self.ax, self.robot_env.obs_list,
                                self.robot_env.rigid_robot, self.state, self.goal)
        plot_robot(self.ax, self.robot_env.rigid_robot, self.state[:3])
        self.ax.axis([-22, 22, -22, 22])
        plt.pause(0.0001)
        return None

    def plot_observation(self):
        wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        tmp_wPos = self.T_transform2d(wTb, self.sample_positions)
        local_map = self.sample_local_map()
        ind_non_free = local_map == 0
        ind_free = local_map > 0

        # plot boundary
        left_bot = [self.local_map_size[2], self.local_map_size[0]]
        left_top = [self.local_map_size[2], self.local_map_size[1]]
        rigit_top = [self.local_map_size[3], self.local_map_size[1]]
        rigit_bot = [self.local_map_size[3], self.local_map_size[0]]
        boundry = np.array([left_bot, left_top, rigit_top,
                            rigit_bot, left_bot], dtype=np.float32)
        boundry = self.T_transform2d(wTb, boundry)
        plt.plot(boundry[:, 0], boundry[:, 1], c='cyan')
        plt.plot(tmp_wPos[ind_free, 0], tmp_wPos[ind_free, 1],
                 '.', c='g', markersize=1)
        plt.plot(tmp_wPos[ind_non_free, 0],
                 tmp_wPos[ind_non_free, 1], '.', c='purple', markersize=1)

#################### collision-unaware version #################


class DubinGymCU(DubinGym):
    def __init__(self):
        super().__init__(obs_list_list=[])
        self.observation_space = self.state_space
        self.max_time = 50.0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = -0.1*info['goal_dis'] + 20.0*info['goal']
        return obs, reward, done, info

    def _obs(self):
        obs = super()._obs()[1] # without local map
        return obs
