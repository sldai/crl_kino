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

from tianshou.data import Batch

class RRT_RL(RRT):
    def __init__(self, robot_env: DifferentialDriveEnv, rl_policy, goal_sample_rate=5, max_iter=500):
        super().__init__(robot_env, goal_sample_rate, max_iter)
        self.policy = rl_policy
        obs = robot_env.obs_list.copy()
        self.gym = DifferentialDriveGym(robot_env)
        self.gym.robot_env.set_obs(obs)

    def steer(self, from_node, to_node, t_max_extend=10.0, t_tree=5.0):
        # using RL policy to steer from_node to to_node
        parent_node = from_node
        new_node = self.Node(from_node.state)
        new_node.parent = parent_node
        new_node.path.append(new_node.state)
        state = new_node.state
        goal = to_node.state.copy()

        env = self.gym
        env.state = state
        env.goal = goal
        obs = env._obs()
        dt = 0.2
        n_max_extend = round(t_max_extend/dt)
        n_tree = round(t_tree/dt)
        new_node_list = []
        
        for n_extend in range(1, n_max_extend+1):
            obs_batch = Batch(obs=obs.reshape((1, -1)), info=None)
            action_batch = self.policy.forward(obs_batch, deterministic=True)
            action = action_batch.act
            action = action.detach().numpy().flatten()
            # control with RL policy
            obs, rewards, done, info = env.step(action)

            state = env.state
            new_node.state = state
            new_node.path.append(state)
            if not env.robot_env.valid_state_check(state):
                # collision
                break
            elif np.linalg.norm(state[:2]-goal[:2]) <= 1.0:
                # reach
                new_node_list.append(new_node)
                break
            elif n_extend % n_tree == 0:
                new_node_list.append(new_node)
                parent_node = new_node
                new_node = self.Node(parent_node.state)
                new_node.parent = parent_node
                new_node.path.append(new_node.state)
                
        self.node_list += new_node_list
        return new_node_list
    
