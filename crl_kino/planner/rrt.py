"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""


import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import time
from crl_kino.env import DifferentialDriveEnv
from crl_kino.policy.dwa import DWA
from crl_kino.utils.draw import *
from abc import ABC, abstractmethod


class RRT(ABC):
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """
        def __init__(self, state: np.ndarray):
            self.state = state.copy()
            self.path = []
            self.parent = None

    def __init__(self, robot_env: DifferentialDriveEnv, goal_sample_rate=5, max_iter=500):
        """
        """
        self.robot_env = robot_env
        self.dwa = DWA(robot_env)
        self.dwa.set_dwa(dt=0.2, v_res=0.02)
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
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
            print('Iteration ' + str(i) +':', str(len(self.node_list))+' nodes')
            good_sample = False
            while not good_sample:
                rnd_node = self.sample(self.goal)
                if not self.robot_env.valid_state_check(rnd_node.state): 
                    continue
                nearest_ind, cost = self.choose_parent(self.node_list, rnd_node)
                if 2.0<= cost < 20: good_sample = True
            nearest_node = self.node_list[nearest_ind]

            new_node_list = self.steer(nearest_node, rnd_node)

            if len(new_node_list)>0:
                if self.reach(self.node_list[-1].state,self.goal.state): # reach goal
                    reach_exactly = True
                    break
                nearest_ind, cost = self.choose_parent(new_node_list, self.goal)
                if cost<8.0: # retry to steer to goal 
                    new_node_list = self.steer(new_node_list[nearest_ind], self.goal)
                    if self.reach(self.node_list[-1].state,self.goal.state):
                        reach_exactly = True
                        break
        path = self.generate_final_course(-1)
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
        # using dwa control to steer from_node to to_node
        parent_node = from_node
        new_node = self.Node(from_node.state)
        new_node.parent = parent_node
        state = new_node.state
        goal = to_node.state.copy()

        dwa = self.dwa
        env = self.robot_env
        dt = 0.2
        n_max_extend = round(t_max_extend/dt)
        n_tree = round(t_tree/dt)
        new_node_list = []
        
        for n_extend in range(1, n_max_extend+1):
            # control with dwa
            v, traj = dwa.control(state, goal) 

            state = env.motion_velocity(state, v, dt)
            new_node.state = state
            new_node.path.append(state)
            if not env.valid_state_check(state):
                # collision
                break
            elif np.linalg.norm(state[:2]-goal[:2]) <= 1.0:
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

    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
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
    def choose_parent(node_list, rnd_node):
        # consider distance as cost
        Q = np.diag([1.0, 1.0, 0.0, 0.0, 0.0])
        dis_list = np.array([RRT.dist(node.state, rnd_node.state) for node in node_list])
        min_ind = np.argmin(dis_list)
        return min_ind, dis_list[min_ind]

    @staticmethod
    def dist(state, goal):
        """
        Check whether the goal is reached
        """
        Q = np.diag([1.0, 1.0, 0.5, 0.0, 0.0])
        d = goal-state
        d[2] = normalize_angle(d[2])
        return np.linalg.norm(Q @ d)

    def collsion_check(self, path):
        check = True
        for state in path:
            if not self.robot_env.valid_state_check(state):
                check = False
                break
        return check        

    @staticmethod
    def reach(state, goal):
        """
        Check whether the goal is reached
        """
        return RRT.dist(state, goal)<=2.0
from crl_kino.env.differential_env import normalize_angle