#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
from differential_env import plot_ob
import pickle
def main(case_num=20, obs_num=7, size=2.5, min_gap=3.0, bounds = np.array([[-20.0, 20.0],[-20.0, 20.0]])):
    obc_list = np.zeros((case_num, obs_num, 2))
    for i in range(case_num):
        for j in range(obs_num):
            while True:
                obc = np.random.uniform(bounds[:, 0], bounds[:, 1])
                if j==0: break
                # check gap
                gap = np.min(np.linalg.norm(obc_list[i, :j, :]-obc, axis=1))-2*size
                if gap>=min_gap: break
            obc_list[i,j] = obc
        ax = plt.gca()
        ax.cla()
        print(obc_list[i])
        
        plot_ob(ax, obc_list[i,:,:], obs_size=size)
        # plt.xlim(bounds[0])
        
        plt.axis('equal')
        plt.ylim(bounds[1])
        # plt.show()
        plt.savefig('obs_image/'+str(i)+'.jpg')
    pickle.dump(obc_list, open('obc_list.pkl', 'wb'))
            
if __name__ == "__main__":
    main()