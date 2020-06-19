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
from crl_kino.env.differential_gym import DifferentailDriveGymLQR
from crl_kino.policy.rl_policy import policy_forward, load_policy
from tianshou.data import Batch

class RRT_RL(RRT):
    def __init__(self, robot_env, policy, goal_sample_rate=5, max_iter=500):
        super().__init__(robot_env, goal_sample_rate, max_iter)
        # obs = robot_env.obs_list.copy()
        self.gym = DifferentailDriveGymLQR()
        self.policy = policy
        # self.gym.robot_env.set_obs(obs)

    def steer(self, from_node, to_node, t_max_extend=10.0, t_tree=5.0):
        # using RL policy to steer from_node to to_node
        x_old = from_node
        x_new = self.Node(x_old.state)
        x_new.parent = x_old
        x_new.path.append(x_new.state)
        
        dt = 0.2
        n_max_extend = round(t_max_extend/dt)
        n_tree = round(t_tree/dt)
        n_step = 0

        
        env = self.gym
        env.state = x_new.state.copy()
        env.goal = to_node.state.copy()
        
        new_node_list = []

        while not (n_step>n_max_extend or self.reach(x_new.state, to_node.state) or not self.robot_env.valid_state_check(x_new.state)):
            obs = env._obs()
            action = policy_forward(self.policy, obs, eps=0.05)[0]
            env.step(action)
            
            x_new.state = env.state.copy()
            x_new.path.append(x_new.state)

            n_step += 1
            if self.robot_env.valid_state_check(x_new.state) and (self.reach(x_new.state, to_node.state) or n_step % n_tree == 0):
                new_node_list.append(x_new)
                x_old = x_new
                x_new = self.Node(x_old.state)
                x_new.parent = x_old
                x_new.path.append(x_new.state)

                
        self.node_list += new_node_list
        return new_node_list
    
