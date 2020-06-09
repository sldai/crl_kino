#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn
import matplotlib.animation as anim

def plot_arrow(x, y, yaw, length=1, width=0.5): 
    return plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            head_length=width, head_width=width, fc='k', ec='k', zorder=0)

def plot_robot(ax, x, y, yaw, robot_size, color):
    mark, = plt.plot(x, y, "x", c=color)
    arrow = plot_arrow(x, y, yaw)
    return [mark, arrow]

def plot_ob(ax, obs_list, obs_size):
    circle_list = []
    for obs in obs_list:
        circle = patches.Circle(obs, obs_size)
        # Add the patch to the Axes
        circle_list.append(ax.add_patch(circle))
    return circle_list


def plot_problem_definition(ax, obs_list, obs_size, robot_size, start, goal):
    """
    plot the obstacles, start and goal 

    Parameters
    ----------
    ax: 
        figure axis
    obs_list: array_like
        list of obstacles
    obs_size: float
        obstacle size
    start: array_like
        start state
    goal: array_like
        goal state
    
    Return
    ------
    collection: list
        a collection of matplotlib artists
    """
    collection = [] 
    ax_ob = plot_ob(ax, obs_list, obs_size)
    start_mark = plot_robot(ax, *start[:3], robot_size, 'r')
    goal_mark = plot_robot(ax, *goal[:3], robot_size, 'b')
    collection += ax_ob + start_mark + goal_mark
    return collection

def draw_path(robot_env, start, goal, path, fname='rrt_path'):
    fig, ax = plt.subplots(figsize=(6,6))
    plt.axis([robot_env.env_bounds[0,0], robot_env.env_bounds[0,1], robot_env.env_bounds[1,0], robot_env.env_bounds[1,1]])
    plt.axis("equal")
    plt.grid(True)
    collection_list = [] # each entry is a collection
    tmp = plot_problem_definition(ax, robot_env.obs_list, robot_env.obs_size, robot_env.robot_radius, start, goal)
    collection_list.append(tmp)

    for state in path:
        tmp_ = tmp.copy()
        robot_marker = plot_robot(ax, *state[:3], robot_env.robot_radius, 'r')
        tmp_ += robot_marker
        collection_list.append(tmp_)
    gif = anim.ArtistAnimation(fig, collection_list, interval=200)
    gif.save(fname+'.gif', writer = anim.PillowWriter(fps=5))