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
        for i in range(self.max_iter):
            good_sample = False
            while not good_sample:
                rnd_node = self.sample(self.goal)
                if not self.robot_env.valid_state_check(rnd_node.state): 
                    continue
                nearest_ind, cost = self.choose_parent(self.node_list, rnd_node)
                if 1.0<= cost < 20: good_sample = True
            nearest_node = self.node_list[nearest_ind]

            new_node_list = self.steer(nearest_node, rnd_node)

            if len(new_node_list)>0:
                if np.linalg.norm(self.node_list[-1].state[:2]-self.goal.state[:2]) <= 1.0: # reach goal
                    break
                nearest_ind, cost = self.choose_parent(new_node_list, self.goal)
                if cost<5.0: # retry to steer to goal 
                    new_node_list = self.steer(new_node_list[nearest_ind], self.goal)
                    if np.linalg.norm(self.node_list[-1].state[:2]-self.goal.state[:2]) <= 1.0:
                        break
        path = self.generate_final_course(-1)
        toc = time.time()
        self.planning_time = toc-tic
        return path


    def set_start_and_goal(self, start: np.ndarray, goal: np.ndarray):
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
        dis_list = np.array([np.linalg.norm(node.state[:2]-rnd_node.state[:2]) for node in node_list])
        min_ind = np.argmin(dis_list)
        return min_ind, dis_list[min_ind]

    def collsion_check(self, path):
        check = True
        for state in path:
            if not self.robot_env.valid_state_check(state):
                check = False
                break
        return check        


    def draw_tree(self, fname='rrt_tree'):
        fig, ax = plt.subplots(figsize=(6,6))
        plt.axis([self.robot_env.env_bounds[0,0], self.robot_env.env_bounds[0,1], self.robot_env.env_bounds[1,0], self.robot_env.env_bounds[1,1]])
        plt.axis("equal")
        plt.grid(True)
        collection_list = [] # each entry is a collection
        tmp = plot_problem_definition(ax, self.robot_env.obs_list, self.robot_env.obs_size, self.robot_env.robot_radius, self.start.state, self.goal.state)
        collection_list.append(tmp)
        for node in self.node_list:
            if node.parent:
                path = np.array(node.path[:])
                ax_path, = plt.plot(path[:,0], path[:,1], "-g")   
                ax_node, = plt.plot(node.state[0], node.state[1], 'x', c='black')
                tmp = tmp.copy()
                tmp.append(ax_path)
                tmp.append(ax_node)
                collection_list.append(tmp)
                # plt.pause(2)   
        gif = anim.ArtistAnimation(fig, collection_list, interval=50)
        gif.save(fname+'.gif', writer = anim.PillowWriter(fps=4))
        # plt.show()  

    def draw_path(self, path, fname='rrt_path'):
        fig, ax = plt.subplots(figsize=(6,6))
        plt.axis([self.robot_env.env_bounds[0,0], self.robot_env.env_bounds[0,1], self.robot_env.env_bounds[1,0], self.robot_env.env_bounds[1,1]])
        plt.axis("equal")
        plt.grid(True)
        collection_list = [] # each entry is a collection
        tmp = plot_problem_definition(ax, self.robot_env.obs_list, self.robot_env.obs_size, self.robot_env.robot_radius, self.start.state, self.goal.state)
        collection_list.append(tmp)

        for state in path:
            tmp_ = tmp.copy()
            robot_marker = plot_robot(ax, *state[:3], self.robot_env.robot_radius, 'r')
            tmp_ += robot_marker
            collection_list.append(tmp_)
        gif = anim.ArtistAnimation(fig, collection_list, interval=200)
        gif.save(fname+'.gif', writer = anim.PillowWriter(fps=5))
