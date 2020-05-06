"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

from data_loader import *
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import time
from differential_env import DifferentialDriveEnv
import seaborn
import matplotlib.animation as anim

class RRT:
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
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.robot_env = robot_env
        self.robot_env.set_dwa(dt=0.2, v_res=0.02)
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.start = None
        self.goal = None
        self.planning_time = 0.0

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
                nearest_ind, cost = self.choose_parent(self.node_list, rnd_node)
                c = self.robot_env.valid_state_check(rnd_node.state) and cost < 20
                if c: good_sample = True
            nearest_node = self.node_list[nearest_ind]

            new_node_list = self.steer_dwa(nearest_node, rnd_node)

            if len(new_node_list)>0:
                nearest_ind, cost = self.choose_parent(new_node_list, self.goal)
                if cost<5.0:  
                    new_node_list = self.steer_dwa(new_node_list[nearest_ind], self.goal)
                    for n in new_node_list:
                        print(np.linalg.norm(n.state[:2]-self.goal.state[:2]))
                    if np.linalg.norm(self.node_list[-1].state[:2]-self.goal.state[:2]) <= 0.3:
                        path = self.generate_final_course(-1)
                        break
        toc = time.time()
        self.planning_time = toc-tic
        return path

    def set_start_and_goal(self, start: np.ndarray, goal: np.ndarray):
        self.start = self.Node(start)
        self.goal = self.Node(goal)

    def steer_dwa(self, from_node, to_node, t_max_extend=10.0, t_tree=5.0):
        # using dwa control to steer from_node to to_node
        parent_node = from_node
        new_node = self.Node(from_node.state)
        new_node.parent = parent_node
        new_node.path.append(new_node.state)
        state = new_node.state
        goal = to_node.state.copy()

        dwa = self.robot_env.dwa
        env = self.robot_env
        dt = dwa.dt
        n_max_extend = round(t_max_extend/dt)
        n_tree = round(t_tree/dt)
        n_extend = 0
        new_node_list = []
        while n_extend <= n_max_extend:
            if not env.valid_state_check(state):
                # collision
                break
            if np.linalg.norm(state[:2]-goal[:2]) <= 2* env.robot_radius:
                # reach
                if n_extend != 0: 
                    self.node_list.append(new_node)
                    new_node_list.append(new_node)
                break
            if n_extend % n_tree == 0 and n_extend != 0:
                self.node_list.append(new_node)
                new_node_list.append(new_node)
                parent_node = new_node
                new_node = self.Node(parent_node.state)
                new_node.parent = parent_node
                new_node.path.append(new_node.state)
            v, traj = dwa.control(state, goal)
            state = env.motion_velocity(state, v, dwa.dt)
            n_extend += 1
            new_node.state = state
            new_node.path.append(state)
            
        return new_node_list

    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.state.copy())
            node = node.parent
        path.append(node.state.copy())
        path.reverse()
        return path

    def sample(self, goal):
        if random.randint(0, 100) > self.goal_sample_rate:
            bounds = self.robot_env.get_bounds()
            state_bounds = bounds['state_bounds']
            state = np.zeros(len(state_bounds))
            for i in range(len(state_bounds)):
                state[i] = np.random.uniform(
                    state_bounds[i, 0], state_bounds[i, 1])
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

    @staticmethod
    def plot_ob(ax, obs_list, obs_size):
        circle_list = []
        for obs in obs_list:
            circle = patches.Circle(obs, obs_size)
            # Add the patch to the Axes
            circle_list.append(ax.add_patch(circle))
        return circle_list

    def draw_tree(self):
        fig, ax = plt.subplots(figsize=(6,6))
        collection_list = [] # each entry is a collection
        ax_ob = self.plot_ob(ax, self.robot_env.obs_list, self.robot_env.obs_size)

        start_mark, = plt.plot(self.start.state[0], self.start.state[1], "or")
        goal_mark, = plt.plot(self.goal.state[0], self.goal.state[1], "or")
        plt.axis([self.robot_env.env_bounds[0,0], self.robot_env.env_bounds[0,1], self.robot_env.env_bounds[1,0], self.robot_env.env_bounds[1,1]])
        plt.axis("equal")
        plt.grid(True)
        tmp = []
        tmp.extend(ax_ob)
        tmp.extend([start_mark, goal_mark])
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
        gif.save('tree.gif', writer = anim.PillowWriter(fps=4))
        plt.show()  
        
        





def main():
    env = DifferentialDriveEnv(2, -0.3, math.pi, 0.7, math.pi / 3 * 5)
    planner = RRT(env)
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
    

    planner.set_start_and_goal(start, goal)
    path = planner.planning()
    planner.draw_tree()

    # obc = load_test_dataset_no_cae()
    # plan_time = []
    # for i in range(len(paths)): 
    #     env.set_obs(obc[i])
    #     for j in range(len(paths[0])):
    #         if path_lengths[i,j]>0:
    #             start = paths[i,j,0,:]
    #             goal = paths[i,j,path_lengths[i,j]-1,:]
    #             planner.set_start_and_goal(start, goal)
    #             if planner.planning() != None:
    #                 plan_time.append(planner.planning_time)
    # np.savetxt('plan_time.csv', np.array(plan_time), delimiter=',')



    
    



 

if __name__ == '__main__':
    main()
