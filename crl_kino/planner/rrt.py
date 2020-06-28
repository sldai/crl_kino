"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""


import random
import numpy as np
import time
from crl_kino.env.dubin_env import DubinEnv
from crl_kino.env.dubin_gym import DubinGymCU
from crl_kino.utils.draw import *
from abc import ABC, abstractmethod
from crl_kino.policy.rl_policy import policy_forward, load_policy

class RRT(ABC):
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """
        def __init__(self, state):
            self.state = state.copy()
            self.path = []
            self.parent = None
            self.cost = 0.0

    def __init__(self, robot_env, policy, ttr_estimator, goal_sample_rate=5, max_iter=500):
        """
        """
        self.robot_env = robot_env
        self.ttr_estimator = ttr_estimator
        self.gym = DubinGymCU()
        self.gym.robot_env = robot_env
        self.policy = policy

        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.epsilon = 10.0

        self.node_list = []
        self.start = None
        self.goal = None
        self.planning_time = 0.0
        self.path = None
        self.state_bounds = self.robot_env.get_bounds()['state_bounds']


    def planning(self):
        """
        rrt path planning
        """
        if self.start is None or self.goal is None:
            raise ValueError('start or goal is not set')
        tic = time.time()
        self.node_list = [self.start]
        path = None
        reach_exactly = False
        for i in range(self.max_iter):
            print('Iteration {}: {} nodes'.format(i, len(self.node_list)))
            good_sample = False
            while not good_sample:
                rnd_node = self.sample(self.goal)
                if not self.robot_env.valid_state_check(rnd_node.state): 
                    continue
                nearest_node, cost = self.nearest(self.node_list, rnd_node, self.Eu_metric)
                if cost < self.epsilon: 
                    good_sample = True
            
            parent_node = self.choose_parent(rnd_node)
            new_node_list = self.steer(parent_node, rnd_node)

            if len(new_node_list)>0:
                if self.nearest(new_node_list, self.goal, self.se2_metric)[1] <= 1.5: # reach goal
                    reach_exactly = True
                    break
                nearest_node, cost = self.nearest(self.node_list, self.goal, self.Eu_metric)
                if cost < 5.0: 
                    parent_node = self.choose_parent(self.goal)
                    new_node_list = self.steer(parent_node, self.goal)
                    if len(new_node_list)>0 and self.nearest(new_node_list, self.goal, self.se2_metric)[1] <= 1.5:
                        reach_exactly = True
                        break
        path = self.generate_final_course(self.node_list[-1])
        toc = time.time()
        self.planning_time = toc-tic
        self.path = path
        self.reach_exactly = reach_exactly
        return path


    def set_start_and_goal(self, start: np.ndarray, goal: np.ndarray):
        assert self.robot_env.valid_state_check(start) and self.robot_env.valid_state_check(goal),\
                'The start or goal states are not valid'
        self.start = self.Node(start)
        self.goal = self.Node(goal)

    
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
            if info['collision']:
                # collision
                break
            elif info['goal']:
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
        self.propagate_cost_to_leaves(from_node)
        return new_node_list

    def generate_final_course(self, node):
        path = []
        while node.parent is not None:
            path = node.path[1:] + path
            node = node.parent
        path = [node.state] + path 
        return path

    def sample(self, goal):
        if random.randint(0, 100) > self.goal_sample_rate:
            state = np.random.uniform(self.state_bounds[:,0], self.state_bounds[:,1])
            rnd = self.Node(state)

        else:  # goal point sampling
            rnd = self.Node(self.goal.state)

        return rnd

    @staticmethod
    def Eu_metric(src_node, dst_node):
        return np.linalg.norm(dst_node.state[:2]-src_node.state[:2])

    @staticmethod
    def se2_metric(src_node, dst_node):
        src_x = np.zeros(4)
        src_x[:2] = src_node.state[:2]
        src_x[2], src_x[3] = np.cos(src_node.state[2]), np.sin(src_node.state[2])

        dst_x = np.zeros(4)
        dst_x[:2] = dst_node.state[:2]
        dst_x[2], dst_x[3] = np.cos(dst_node.state[2]), np.sin(dst_node.state[2])
        return np.linalg.norm(dst_x - src_x)

    def ttr_metric(self, src_node, dst_node):
        self.gym.state = src_node.state.copy()
        self.gym.goal = dst_node.state.copy()
        obs = self.gym._obs().reshape(1,-1)
        ttr = self.ttr_estimator(obs).item()
        return ttr

    @staticmethod
    def near(node_list, rnd_node, metric, delta):
        dis_list = np.array([metric(node, rnd_node) for node in node_list])
        ind = np.flatnonzero(dis_list<=delta)
        if len(ind)==0: 
            return None
        else:
            return [node_list[i] for i in ind]
   
    @staticmethod
    def nearest(node_list, rnd_node, metric):
        dis_list = np.array([RRT.se2_metric(node, rnd_node) for node in node_list])
        min_ind = np.argmin(dis_list)
        return node_list[min_ind], dis_list[min_ind]


    def choose_parent(self, rnd_node):
        """
        The chosen node has the lowest heuristic
        """
        near_nodes = self.near(self.node_list, rnd_node, self.Eu_metric, self.epsilon)
        parent_node = None
        if near_nodes == None:
            parent_node, min_c_cost = self.nearest(self.node_list, rnd_node, self.Eu_metric)
        else:
            parent_node, min_h_cost = self.nearest(near_nodes, rnd_node, self.ttr_metric)
        return parent_node

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost+0.2*(len(node.path)-1) 
                self.propagate_cost_to_leaves(node)
  


if __name__ == "__main__":
    import pickle, os
    from crl_kino.estimator.network import TTRCU
    import torch
    from crl_kino.policy.rl_policy import policy_forward, load_policy
    # test_env1 = pickle.load(open('data/obstacles/test_env1.pkl', 'rb'))
    # start = np.array([-5, -15, 0, 0, 0.0])
    # goal = np.array([10, 10, 0, 0, 0.0])

    test_env2 = pickle.load(open('data/obstacles/test_env2.pkl', 'rb'))
    start = np.array([-15.0,17,0])
    goal = np.array([10.8,-8.5,0])

    # robot_env
    env, gym = DubinEnv(), DubinGymCU()
    env.set_obs(test_env2)

    # policy
    model_path = os.path.join('log', 'dubin', 'ddpg/policy.pth')
    policy = load_policy(gym, [1024, 512, 512, 512], model_path)

    # ttr_estimator
    estimator = TTRCU(gym.observation_space.shape[0], 1, 'cpu').to('cpu')
    estimator.load_state_dict(torch.load('log/estimator/estimator.pth', map_location='cpu'))

    planner = RRT(env, policy=policy, ttr_estimator=estimator, max_iter=1000)

    planner.set_start_and_goal(start, goal)

    path = planner.planning()
    draw_path(env, start, goal, path)
    draw_tree(env, start, goal, planner.node_list)
    pickle.dump(planner, open('rrt.pkl','wb'))