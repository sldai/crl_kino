#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path
from crl_kino.env import DifferentialDriveEnv, DifferentialDriveGym
from crl_kino.policy.dwa import DWA
from crl_kino.utils.draw import *
from crl_kino.planner.rrt import RRT
from crl_kino.planner.rrt_rl import RRT_RL
from crl_kino.policy.rl_policy import load_RL_policy
from crl_kino.planner.sst import SST
import argparse


def main():
    repeat = 10
    env = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)

    obs = np.array([[-10.402568,   -5.5128484],
                    [14.448388,   -4.1362205],
                    [10.003768,   -1.2370133],
                    [11.609167,    0.9119211],
                    [-4.9821305,   3.8099794],
                    [8.94005,    -4.14619],
                    [-10.45487,     6.000557]])
    env.set_obs(obs)

    # collect data of rl_rrt
    policy = load_RL_policy([1024, 768, 512], os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data/log/mid_noise/ddpg/policy.pth'))
    rl_rrt = RRT_RL(env, policy)
    start = np.array([13, -7.5, 0, 0, 0.0])
    goal = np.array([10, 10, 0, 0, 0.0])
    rl_rrt.set_start_and_goal(start, goal)
    rl_rrt_path_len = np.zeros(repeat)
    rl_rrt_runtime = np.zeros(repeat)
    for i in range(repeat):
        path = rl_rrt.planning()
        rl_rrt_path_len[i] = 0.2*(len(path)-1)
        rl_rrt_runtime[i] = rl_rrt.planning_time

    # collect data of rrt
    dwa_rrt = RRT(env)
    start = np.array([13, -7.5, 0, 0, 0.0])
    goal = np.array([10, 10, 0, 0, 0.0])
    dwa_rrt.set_start_and_goal(start, goal)
    dwa_rrt_path_len = np.zeros(repeat)
    dwa_rrt_runtime = np.zeros(repeat)
    for i in range(repeat):
        path = dwa_rrt.planning()
        dwa_rrt_path_len[i] = 0.2*(len(path)-1)
        dwa_rrt_runtime[i] = dwa_rrt.planning_time

    # collect data of sst
    sst = SST(env)
    sst.set_start_and_goal(start, goal)
    sst_path_len = np.zeros(repeat)
    sst_runtime = np.zeros(repeat)
    for i in range(repeat):
        check = sst.planning()
        sst_path_len[i] = np.sum(sst.path[:, -1])
        sst_runtime[i] = sst.planning_time

    data = {
        'rl_rrt': {'path_len': rl_rrt_path_len, 'runtime': rl_rrt_runtime},
        'dwa_rrt': {'path_len': dwa_rrt_path_len, 'runtime': dwa_rrt_runtime},
        'sst': {'path_len': sst_path_len, 'runtime': sst_runtime},
    }
    pickle.dump(data, open('data.pkl', 'wb'))
    # fig, ax = plt.subplots()
    # ax.plot(rl_rrt_path_len, label='RL-RRT')
    # ax.plot(dwa_rrt_path_len, label='DWA-RRT')
    # ax.plot(sst_path_len, label='SST')
    # plt.xlabel('')


if __name__ == "__main__":
    main()
