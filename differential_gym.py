#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
from data_loader import load_test_dataset_no_cae
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from differential_env import DifferentialDriveEnv, normalize_angle
import itertools
import transforms3d.euler as euler

class DifferentialDriveGym(gym.Env):
    """
    Custom Environment that follows gym interface
    """

    def __init__(self, robot_env: DifferentialDriveEnv = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)):
        """
        :param robot_env: simulation environment
        :param curriculum: difficulty of env, obs_num specifies #obstacle in env, ori makes the robot heads to the goal at initialization
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
        self.action_space = spaces.Box(
            low=0, high=1, shape=self.v_space.shape)

        self.sample_positions = self._init_sample_positions()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(
            len(self.sample_positions)+4,))

        self.n_step = 0
        self.max_step = 200
        self.current_time = 0.0
        self.max_time = 100.0

        self.obc_list = self.init_training_envs()
        self.n_case = 0

        self.curriculum = {'obs_num': 0, 'ori': True}
        self.reset()

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
        local_map = [1.0*self.robot_env.valid_point_check(pos) for pos in tmp_wPos]
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
        ind_free = local_map >0

        # plot boundary
        left_bot = [self.percept_region[2], self.percept_region[0]]
        left_top = [self.percept_region[2], self.percept_region[1]]
        rigit_top = [self.percept_region[3], self.percept_region[1]]
        rigit_bot = [self.percept_region[3], self.percept_region[0]]
        boundry = np.array([left_bot, left_top, rigit_top, rigit_bot, left_bot], dtype = np.float32)
        boundry = self.T_transform2d(wTb, boundry)
        plt.plot(boundry[:,0], boundry[:,1], c='cyan')
        plt.plot(tmp_wPos[ind_free, 0], tmp_wPos[ind_free, 1], '.', c = 'g', markersize = 1)
        plt.plot(tmp_wPos[ind_non_free, 0], tmp_wPos[ind_non_free, 1], '.', c = 'black', markersize = 1)

    def a2v(self, action):
        '''
        action space to v space
        '''
        v = np.zeros(self.action_space.shape)
        for i in range(len(action)):
            v[i] = (action[i]-self.action_space.low[i])/(self.action_space.high[i]-self.action_space.low[i]) * (self.v_space.high[i]-self.v_space.low[i]) + self.v_space.low[i]
        return v
    def v2a(self, v):
        '''
        v space to action space
        '''
        action = np.zeros(self.action_space.shape)
        for i in range(len(v)):
            action[i] = (v[i]-self.v_space.low[i])/(self.v_space.high[i]-self.v_space.low[i]) * (self.action_space.high[i]-self.action_space.low[i]) + self.action_space.low[i]
        return action

    def step(self, action):
        v = self.a2v(action)
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
                'collision': False,
                'clearance': 0.0,
                'speed': self.state[3],
                'step': 1.0
                }
        info['goal_dis'] = np.linalg.norm(self.state[:2]-self.goal[:2])
        if info['goal_dis'] <= self.robot_env.robot_radius*2:
            info['goal'] = True

        info['clearance'] = self.robot_env.get_clearance(self.state)
        info['collision'] = not self.robot_env.valid_state_check(self.state)
        # info['step'] = self.n_step
        return info
    
    def _reward(self, info):
        info_arr = np.array([info[i] for i in info])
        reward = info_arr @ np.array([150.0, -0.48, -200.0, 0, 0.01, -0])
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
        if len(bP.shape) == 1: # vector
            bP_ = np.concatenate((bP, np.ones(1)))
            aP_ = aTb @ bP_
            aP = aP_[:2]
        elif bP.shape[1] == 2:
            bP = bP.T
            bP_ = np.vstack((bP, np.ones((1,bP.shape[1]))))
            aP_ = aTb @ bP_
            aP = aP_[:2,:].T
        return aP


    def init_training_envs(self):
        obc = load_test_dataset_no_cae()
        return obc

    def set_curriculum(self, **kwargs): 
        for k, v in kwargs.items():
            if self.curriculum[k] is not None:
                self.curriculum[k] = v
    def reset(self):
        ind_obs = np.random.randint(0, len(self.obc_list))
        assert 0<=self.curriculum['obs_num']<=len(self.obc_list[ind_obs])
        self.robot_env.set_obs(self.obc_list[ind_obs][:self.curriculum['obs_num']])
        
        # sample a random start goal configuration
        start = np.zeros(len(self.state_bounds))
        goal = np.zeros(len(self.state_bounds)) 
        while True:
            # random sample start and goal configuration
            start[:3] = np.random.uniform(self.state_bounds[:3, 0], self.state_bounds[:3, 1])
            goal[:3] = np.random.uniform(self.state_bounds[:3, 0], self.state_bounds[:3, 1])
            
            # start point to goal
            if self.curriculum['ori']: start[2] = normalize_angle(np.arctan2(goal[1]-start[1],goal[0]-start[0]))
            if self.robot_env.get_clearance(start)>0.1 and self.robot_env.get_clearance(goal)>0.1 and 5.0<np.linalg.norm(start[:2]-goal[:2])<10.0:
                break
        
        self.state = start
        self.goal = goal

        self.current_time = 0
        obs = self._obs()
        return obs

    def render(self, mode='human', plot_localwindow = True): 
        ax = plt.gca()
        ax.cla() # clear things 
        
        self.robot_env.plot_ob(ax, self.robot_env.obs_list, self.robot_env.obs_size)
        if plot_localwindow:
            self.plot_observation()
        plt.plot(self.state[0], self.state[1], "xr")
        plt.plot(self.goal[0], self.goal[1], "xb")
        self.robot_env.plot_arrow(*self.state[:3], length=1, width=0.5)
        
        plt.axis('equal')
        plt.ylim(-20.0, 20.0)
        plt.xlim(-30.0, 30.0)
        plt.pause(0.0001)

        return np.array([[[1,1,1]]
                         ], dtype=np.uint8)


def dwa_control_gym():
    '''
    debug gym
    '''
    env = DifferentialDriveGym()
    env.set_curriculum(ori = False,obs_num =7)
    print(env.action_space.shape)
    print(env.action_space.high[0])
    env.reset()
    start = env.state
    state = start.copy()
    goal = env.goal
    print(np.linalg.norm(goal[:2]-start[:2]))
    env.robot_env.set_dwa(dt = 0.3)
    dwa = env.robot_env.dwa
    
    fig, ax = plt.subplots()
    for i in range(200):
        v, traj = dwa.control(state, goal)
        print(v)
        obs, reward, done, info = env.step(env.v2a(v))
        # print(reward)
        # print(obs[:4])
        state = env.state
        if done: print(done)
        if info['collision']: 
            print('Collision')
            break
        if info['goal']: 
            print('Goal')
            break
        
        env.render(plot_localwindow=True)
    plt.show()


if __name__ == "__main__":
    dwa_control_gym()
