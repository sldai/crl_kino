#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class RobotEnv(object):
    @abstractmethod
    def motion(self): pass
    # motion model of the robot

    @abstractmethod
    def valid_state_check(self): pass
    # collision detect

    @abstractmethod
    def get_bounds(self): pass
    # return bounds of states, controls
