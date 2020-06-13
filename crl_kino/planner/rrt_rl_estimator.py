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
import torch


class RRT_RL_Estimator(RRT):
    def __init__(self, robot_env, policy, estimator, classifier, goal_sample_rate=5, max_iter=500):
        super().__init__(robot_env, goal_sample_rate, max_iter)
        # obs = robot_env.obs_list.copy()
        self.gym = DifferentialDriveGym(robot_env)
        self.policy = policy
        self.estimator = estimator
        self.classifier = classifier
        # self.gym.robot_env.set_obs(obs)

    def choose_parent(self, node_list, rnd_node):
        
        node_list_valid = []
        for idx, node in enumerate(node_list):
            if 1.0 <= np.linalg.norm(node.state[:2]-rnd_node.state[:2]) < 20.0:
                node_list_valid.append(idx)

        if len(node_list_valid) == 0:
            return 0, 100


        to_node = rnd_node
        obs_list = []
        env = self.gym
        for idx in node_list_valid:

            parent_node = node_list[idx]
            new_node = self.Node(node_list[idx].state)
            """
            new_node.parent = parent_node
            new_node.path.append(new_node.state)
            """
            state = new_node.state
            goal = to_node.state.copy()

            env.state = state
            env.goal = goal
            obs = env._obs()

            obs_list.append(obs)
        
        inputs = torch.tensor(np.array(obs_list), dtype=torch.float32)
        
        inputs[:, 1] = torch.norm(inputs[:, :2], dim=1)

        spatial = inputs[:, 4:].view([-1, 1, 27, 27])
        ext = inputs[:, 1:4]

        outputs_e = self.estimator(spatial, ext)[:, 0]
        outputs_c = self.classifier(spatial, ext)[:, 0]


        prob = outputs_c.detach().numpy()
        estimate_dist = outputs_e.detach().numpy()
        fail = 1-np.round(prob)
        #print(fail)
        dis_list = np.array([np.linalg.norm(node.state[:2]-rnd_node.state[:2]) for node in node_list])
        dis_list[np.array(node_list_valid)] = (estimate_dist+1)*15
        dis_list[np.array(node_list_valid)] += 100*fail

        min_ind = np.argmin(dis_list)

        #print(min_ind, dis_list[min_ind])
        return min_ind, dis_list[min_ind]

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
            action = policy_forward(self.policy, obs, eps=0.05)[0]
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
    
