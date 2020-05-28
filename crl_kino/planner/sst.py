#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt

from crl_kino.env import DifferentialDriveEnv

import time
from os.path import abspath, dirname, join
import sys
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import control as oc
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    sys.path.insert(
        0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import control as oc
from functools import partial

class SST(object):
    def __init__(self, robot_env: DifferentialDriveEnv):
        self.robot_env = robot_env
        bounds = self.robot_env.get_bounds()
        self.state_bounds = bounds['state_bounds']
        self.control_bounds = bounds['control_bounds']

        # add the state space
        space = ob.RealVectorStateSpace(len(self.state_bounds))
        bounds = ob.RealVectorBounds(len(self.state_bounds))
        # add state bounds
        for k, v in enumerate(self.state_bounds):
            bounds.setLow(k, float(v[0]))
            bounds.setHigh(k, float(v[1]))
        space.setBounds(bounds)
        self.space = space
        # add the control space
        cspace = oc.RealVectorControlSpace(space, len(self.control_bounds))
        # set the bounds for the control space
        cbounds = ob.RealVectorBounds(len(self.control_bounds))
        for k, v in enumerate(self.control_bounds):
            cbounds.setLow(k, float(v[0]))
            cbounds.setHigh(k, float(v[1]))
        cspace.setBounds(cbounds)
        self.cspace = cspace
        # define a simple setup class
        self.ss = oc.SimpleSetup(cspace)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn( \
            partial(self.isStateValid, self.ss.getSpaceInformation())))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))
        
        # set the planner
        si = self.ss.getSpaceInformation()
        planner = oc.SST(si)
        self.ss.setPlanner(planner)
        si.setPropagationStepSize(.1)

        self.start = None
        self.goal = None
        self.planning_time = 0.0
        self.path = None

    @staticmethod
    def obRealVector2array(realvector, n):
        array = np.zeros(n)
        for i in range(n):
            array[i] = realvector[i]
        return array

    def isStateValid(self, spaceInformation, state):
        state_array = self.obRealVector2array(state, len(self.state_bounds))
        return self.robot_env.valid_state_check(state_array) \
               and spaceInformation.satisfiesBounds(state)
    
    def propagate(self, start, control, duration, state):
        state_array = self.obRealVector2array(start, len(self.state_bounds))
        control_array = self.obRealVector2array(control, len(self.control_bounds))
        state_array = self.robot_env.motion(state_array, control_array, duration)
        for i in range(len(state_array)):
            state[i] = state_array[i]
    
    def set_start_and_goal(self, start: np.ndarray, goal: np.ndarray):
        self.start = ob.state(self.space)
        self.goal = ob.state(self.space)

        for i in range(len(self.state_bounds)):
            self.start[i] = start[i]
            self.goal[i] = goal[i]        

    def planning(self):
        if self.start is None and self.goal is None:
            return False
        self.ss.clear()
        tic = time.time()
        self.ss.setStartAndGoalStates(self.start, self.goal, 0.05)
        self.ss.solve(20.0)
        toc = time.time()
        print("Found solution:\n%s" % self.ss.getSolutionPath().printAsMatrix())

def test_sst():
    env = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)

    obs = np.array([[-10.402568,   -5.5128484],
                    [14.448388,   -4.1362205],
                    [10.003768,   -1.2370133],
                    [11.609167,    0.9119211],
                    [-4.9821305,   3.8099794],
                    [8.94005,    -4.14619],
                    [-10.45487,     6.000557]])
    env.set_obs(obs)

    sst = SST(env)
    start = np.array([13, -7.5, 0, 0, 0.0])
    goal = np.array([10, 10, 0, 0, 0.0])

    sst.set_start_and_goal(start, goal)
    sst.planning()

if __name__ == "__main__":
    test_sst()