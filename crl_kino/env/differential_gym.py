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
from crl_kino.env.differential_env import DifferentialDriveEnv, normalize_angle
import itertools
import transforms3d.euler as euler
import pickle
from crl_kino.utils.draw import *
from tianshou.data import Batch

class DifferentialDriveGym(gym.Env):
    """
    The gym wrapper for DifferentailDriveEnv
    """

    def __init__(self,
                 robot_env=DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi),
                 reward_param=np.array(
                     [50.0, -0.5, -2.0, -30.0, 1.0, 1.0, -.5, -2.5]),
                 obc_list=np.zeros((10, 7, 2))):
        """
        :param robot_env: simulation environment
        """
        super(DifferentialDriveGym, self).__init__()
        #
        self.robot_env = robot_env
        self.state_bounds = self.robot_env.get_bounds()['state_bounds'].copy()

        self.state = np.zeros(len(self.state_bounds))
        self.goal = np.zeros(len(self.state))

        # actual range of v, w
        self.v_space = spaces.Box(
            low=self.state_bounds[-2:, 0], high=self.state_bounds[-2:, 1])
        # output of network, need to be scaled to v_space
        self.action_space = self.v_space

        self.sample_positions = self._init_sample_positions()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(
            len(self.sample_positions)+4,))

        self.n_step = 0
        self.max_step = 200
        self.current_time = 0.0
        self.max_time = 100.0

        self.obc_list = obc_list.copy()
        self.n_case = 0

        self.reward_param = reward_param.copy()
        self.curriculum = {'obs_num': 7, 'ori': False}
        # self.reset()

    def _init_sample_positions(self):
        """
        the robot sense the local environment by sampling the local map.
        sample_positions contain the positions need to be sampled in the body coordinate system.
        """
        percept_region = np.array([-4, 4, -2, 6])  # left, right, behind, ahead
        sample_reso = 0.3
        lr = np.arange(percept_region[0], percept_region[1], sample_reso)
        ba = np.arange(percept_region[2], percept_region[3], sample_reso)
        sample_positions = list(itertools.product(ba, lr))
        self.percept_region = percept_region
        self.percept_shape = (1, len(ba), len(lr))  # (channel, width, height)
        self.sample_reso = sample_reso
        return np.array(sample_positions)

    def sample_local_map(self, wTb):
        """
        sampling points of local map, with value representing occupancy (0 is non-free, 255 is free)
        wTb: world transform robot
        """
        # world sample position
        tmp_wPos = self.T_transform2d(wTb, self.sample_positions)
        local_map = np.zeros(len(self.sample_positions))
        local_map = [
            1.0*self.robot_env.valid_point_check(pos) for pos in tmp_wPos]
        # for ind, pos in enumerate(tmp_wPos):
        #     if self.robot_env.valid_point_check(pos):
        #         local_map[ind] = 255
        #     else:
        #         local_map[ind] = 0
        return np.array(local_map)

    def plot_observation(self):
        wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        tmp_wPos = self.T_transform2d(wTb, self.sample_positions)
        local_map = self.sample_local_map(wTb)
        ind_non_free = local_map == 0
        ind_free = local_map > 0

        # plot boundary
        left_bot = [self.percept_region[2], self.percept_region[0]]
        left_top = [self.percept_region[2], self.percept_region[1]]
        rigit_top = [self.percept_region[3], self.percept_region[1]]
        rigit_bot = [self.percept_region[3], self.percept_region[0]]
        boundry = np.array([left_bot, left_top, rigit_top,
                            rigit_bot, left_bot], dtype=np.float32)
        boundry = self.T_transform2d(wTb, boundry)
        plt.plot(boundry[:, 0], boundry[:, 1], c='cyan')
        plt.plot(tmp_wPos[ind_free, 0], tmp_wPos[ind_free, 1],
                 '.', c='g', markersize=1)
        plt.plot(tmp_wPos[ind_non_free, 0],
                 tmp_wPos[ind_non_free, 1], '.', c='black', markersize=1)

    def a2v(self, action):
        '''
        action space to v space
        '''
        v = np.zeros(self.action_space.shape)
        for i in range(len(action)):
            v[i] = (action[i]-self.action_space.low[i])/(self.action_space.high[i] -
                                                         self.action_space.low[i]) * (self.v_space.high[i]-self.v_space.low[i]) + self.v_space.low[i]
        return v

    def v2a(self, v):
        '''
        v space to action space
        '''
        action = np.zeros(self.action_space.shape)
        for i in range(len(v)):
            action[i] = (v[i]-self.v_space.low[i])/(self.v_space.high[i]-self.v_space.low[i]) * (
                self.action_space.high[i]-self.action_space.low[i]) + self.action_space.low[i]
        return action

    def step(self, action):
        v = action
        dt = 1.0/5.0  # 5 Hz
        self.state = self.robot_env.motion_velocity(self.state, v, dt)
        self.current_time += dt
        obs = self._obs()
        info = self._info()
        reward = self._reward(info)
        done = info['goal'] or info['collision'] or self.current_time > self.max_time
        return obs, reward, done, info

    def _info(self):
        info = {'goal': False,
                'goal_dis': 0.0,
                'heading': 0.0,
                'collision': False,
                'clearance': 0.0,
                'v': self.state[3],
                'w': abs(self.state[4]),
                'step': 1.0
                }
        info['goal_dis'] = np.linalg.norm(self.state[:2]-self.goal[:2])
        if info['goal_dis'] <= 1.0:
            info['goal'] = True
        info['heading'] = np.abs(normalize_angle(np.arctan2(
            self.goal[1]-self.state[1], self.goal[0]-self.state[0]) - self.state[2]))
        info['clearance'] = min(self.robot_env.get_clearance(self.state), 1.0)
        info['collision'] = not self.robot_env.valid_state_check(self.state)
        return info

    def _reward(self, info):
        info_arr = np.array([info[i] for i in info])
        reward = info_arr @ self.reward_param
        return reward

    def _obs(self):
        wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])

        local_map = self.sample_local_map(wTb)
        bTw = np.linalg.inv(wTb)
        b_goal_pos = self.T_transform2d(bTw, self.goal[:2])
        # goal position in the robot coordinate, robot velocity, local map
        obs = np.block([b_goal_pos, self.state[-2:], local_map])
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

    def set_curriculum(self, **kwargs):
        for k, v in kwargs.items():
            if self.curriculum[k] is not None:
                self.curriculum[k] = v

    def reset(self):
        ind_obs = np.random.randint(0, len(self.obc_list))
        assert 0 <= self.curriculum['obs_num'] <= len(self.obc_list[ind_obs])
        self.robot_env.set_obs(
            self.obc_list[ind_obs][:self.curriculum['obs_num']])

        # sample a random start goal configuration
        start = np.zeros(len(self.state_bounds))
        goal = np.zeros(len(self.state_bounds))
        while True:
            # random sample start and goal configuration

            # sample a valid state
            start[:] = np.random.uniform(
                self.state_bounds[:, 0], self.state_bounds[:, 1])
            if self.robot_env.get_clearance(start) <= 0.5:
                continue

            # sample a valid goal
            for _ in range(5):
                r = np.random.uniform(2.0, 10.0)
                theta = np.random.uniform(-np.pi, np.pi)
                goal[0] = np.clip(start[0] + r*np.cos(theta), *self.state_bounds[0,:])
                goal[1] = np.clip(start[1] + r*np.sin(theta), *self.state_bounds[1,:])
                if self.robot_env.get_clearance(goal) >= 0.5:
                    break

            # start point to goal
            if self.curriculum['ori']:
                start[2] = normalize_angle(np.arctan2(
                    goal[1]-start[1], goal[0]-start[0]))
            if self.robot_env.get_clearance(start) > 0.5 and self.robot_env.get_clearance(goal) > 0.5 and 2.0 < np.linalg.norm(start[:2]-goal[:2]) < 10.0:
                break

        self.state = start
        self.goal = goal

        self.current_time = 0
        obs = self._obs()
        return obs

    def render(self, mode='human', plot_localwindow=True, pause=True):
        ax = plt.gca()
        ax.cla()  # clear things

        plot_ob(ax, self.robot_env.obs_list, self.robot_env.obs_size)
        if plot_localwindow:
            self.plot_observation()
        plot_robot(ax, *self.state[:3], self.robot_env.robot_radius, 'r')
        plot_robot(ax, *self.goal[:3], self.robot_env.robot_radius, 'b')

        plt.axis('equal')
        plt.ylim(-20.0, 20.0)
        plt.xlim(-30.0, 30.0)
        if pause:
            plt.pause(0.0001)

        return np.array([[[1, 1, 1]]
                         ], dtype=np.uint8)


################### motion primitives ######################
class DifferentialDriveGymPrimitive(DifferentialDriveGym):
    def __init__(self,
                 robot_env=DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi),
                 reward_param=np.zeros(5),
                 obc_list=np.zeros((10, 0, 2))):
        super().__init__(robot_env, reward_param, obc_list)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(
            3,))
        self.max_time = 10.0
        
    def step(self, action):
        v = action
        dt = 1.0/5.0  # 5 Hz
        self.state = self.robot_env.motion_velocity(self.state, v, dt)
        self.current_time += dt
        obs = self._obs()
        info = self._info()
        reward = self._reward()
        done = self.current_time > self.max_time
        return obs, reward, done, info

    def _reward(self):
        vx = self.state[3] * np.cos(self.state[2])
        vy = self.state[3] * np.sin(self.state[2])
        w = self.state[4]
        reward = vx - 0.2*abs(vy) - 1.0*abs(np.tanh(w))
        return reward

    def _obs(self):
        obs = self.state[2:]
        return obs

    def reset(self):
        # sample a random start goal configuration
        start = np.random.uniform(
                self.state_bounds[:, 0], self.state_bounds[:, 1])
        self.state = start
        self.current_time = 0
        obs = self._obs()
        return obs

class DifferentialDriveGymForward(DifferentialDriveGymPrimitive):
    def _reward(self):
        vx = self.state[3] * np.cos(self.state[2])
        vy = self.state[3] * np.sin(self.state[2])
        w = self.state[4]
        reward = vx - 0.5*abs(vy) - 1.0*abs(np.tanh(w))
        return reward

class DifferentialDriveGymBackward(DifferentialDriveGymPrimitive):
    def _reward(self):
        vx = self.state[3] * np.cos(self.state[2])
        vy = self.state[3] * np.sin(self.state[2])
        w = self.state[4]
        reward = -vx - 0.5*abs(vy) - 1.0*abs(np.tanh(w))
        return reward

class DifferentialDriveGymUpward(DifferentialDriveGymPrimitive):
    def _reward(self):
        vx = self.state[3] * np.cos(self.state[2])
        vy = self.state[3] * np.sin(self.state[2])
        w = self.state[4]
        reward = vy - 0.5*abs(vx) - 1.0*abs(np.tanh(w))
        return reward

class DifferentialDriveGymDownward(DifferentialDriveGymPrimitive):
    def _reward(self):
        vx = self.state[3] * np.cos(self.state[2])
        vy = self.state[3] * np.sin(self.state[2])
        w = self.state[4]

        reward = -vy - 0.5*abs(vx) - 1.0*abs(np.tanh(w))
        return reward



####################### compositional policy #####################
class DifferentialDriveGymCompose(DifferentialDriveGym):
    def __init__(self, base_policies, obc_list):
        super().__init__(obc_list=obc_list)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(
            5+len(self.sample_positions)+8,))
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(len(base_policies),))
        self.base_policies = base_policies
        
        
    def primitives(self):
        obs_proprioceptive = Batch(obs=self.state[2:].reshape((1, -1)), info=None)

        v_list = []
        for k,v_ in enumerate(self.base_policies):
            act_batch = v_(obs_proprioceptive, eps=0.05)  
            act = act_batch.act.detach().cpu().numpy()[0]           
            v_primitive = act
            assert len(v_primitive)==2
            v_list.append(v_primitive)
        return v_list
        
    def step(self, action):
        action /= np.sum(action)
        assert abs(np.sum(action) - 1.0) < 1e-5,\
            'the summation of weights'+str(np.sum(action))+'does not equal to 1'
        weights = action
        v_list = self.primitives()

        v = np.array(v_list).T @ weights
        dt = 1.0/5.0  # 5 Hz

        self.state = self.robot_env.motion_velocity(self.state, v, dt)
        self.current_time += dt
        obs = self._obs()
        info = self._info()
        reward = self._reward(info)
        done = info['goal'] or info['collision'] or self.current_time > self.max_time
        return obs, reward, done, info

    def _obs(self):
        wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])

        local_map = self.sample_local_map(wTb)
        obs = np.block([self.state[2:], self.goal[:2]-self.state[:2], local_map]+self.primitives())
        return obs

