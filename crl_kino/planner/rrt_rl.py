#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
from crl_kino.planner.rrt import RRT
from crl_kino.env import DifferentialDriveGym, DifferentialDriveEnv
from crl_kino.policy.rl_policy import policy_forward, load_policy
from tianshou.data import Batch

class RRT_RL(RRT):
    def __init__(self, robot_env, policy, goal_sample_rate=5, max_iter=500):
        super().__init__(robot_env, goal_sample_rate, max_iter)
        # obs = robot_env.obs_list.copy()
        self.gym = DifferentialDriveGym(robot_env)
        self.policy = policy
        # self.gym.robot_env.set_obs(obs)

    
    
