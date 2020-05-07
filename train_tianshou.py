#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
from differential_gym import DifferentialDriveGym

env = DifferentialDriveGym()
print(env.action_space.high[1])