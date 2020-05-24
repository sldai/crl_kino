#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
from rrt import RRT
from differential_gym import DifferentialDriveGym
from differential_env import DifferentialDriveEnv
import torch
from tianshou.policy import DDPGPolicy
from differential_gym import DifferentialDriveGym
from net import Actor, Critic
from tianshou.data import Collector, ReplayBuffer, Batch

class RRT_RL(RRT):
    def __init__(self, robot_env: DifferentialDriveEnv, goal_sample_rate=5, max_iter=500):
        super().__init__(robot_env, goal_sample_rate, max_iter)
        self.policy = load_RL_policy([1024, 768, 512], 'log/mid_noise/ddpg/policy.pth')
        obs = robot_env.obs_list.copy()
        self.gym = DifferentialDriveGym(robot_env)
        self.gym.robot_env.set_obs(obs)

    def steer(self, from_node, to_node, t_max_extend=10.0, t_tree=5.0):
        # using dwa control to steer from_node to to_node
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
                self.node_list.append(new_node)
                new_node_list.append(new_node)
                break

            elif n_extend % n_tree == 0:
                self.node_list.append(new_node)
                new_node_list.append(new_node)
                parent_node = new_node
                new_node = self.Node(parent_node.state)
                new_node.parent = parent_node
                new_node.path.append(new_node.state)
   
        return new_node_list
    


def load_RL_policy(layer, model_path):
        device = 'cpu'
        torch.set_num_threads(1)  # we just need only one thread for NN
        env = DifferentialDriveGym()
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        max_action = env.action_space.high[0]

        # model
        actor = Actor(
            layer, state_shape, action_shape,
            max_action, device
        ).to(device)
        actor_optim = torch.optim.Adam(actor.parameters())
        critic = Critic(
            layer, state_shape, action_shape, device
        ).to(device)
        critic_optim = torch.optim.Adam(critic.parameters())
        policy = DDPGPolicy(
            actor, actor_optim, critic, critic_optim,
            action_range=[env.action_space.low[0], env.action_space.high[0]])
        policy.load_state_dict(torch.load(model_path, map_location=device))
        return policy

def main():
    env = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)
    
    obs = np.array([[-10.402568,   -5.5128484],
    [ 14.448388,   -4.1362205],
    [ 10.003768,   -1.2370133],
    [ 11.609167,    0.9119211],
    [ -4.9821305,   3.8099794],
    [  8.94005,    -4.14619  ],
    [-10.45487,     6.000557 ]])
    env.set_obs(obs)
    start = np.array([13, -7.5, 0, 0, 0.0])
    goal = np.array([10, 10, 0, 0, 0.0])
    planner = RRT_RL(env)

    planner.set_start_and_goal(start, goal)
    path = planner.planning()
    planner.draw_path(path)
    planner.draw_tree()

if __name__ == "__main__":
    main()