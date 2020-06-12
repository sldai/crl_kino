import os.path, os
import pickle
from crl_kino.utils.draw import *

from crl_kino.utils.rigid import RectObs, RectRobot, CircleRobot, aabb_intersect_aabb
import matplotlib as mpl

def obs_generate(case_num=10, obs_num_max=8, size_range=[3.0, 20.0], min_gap=3.0, bounds=np.array([[-20.0, 20.0], [-20.0, 20.0]]), dir_path='obstacles'):
    """
    A tool for generating obstacles 
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(6,6))
    color = (122/255.0,110/255.0,119/255.0)
    botboundary = RectObs(np.array([[-22, -22],[22, -20.0]]), color=color)
    topboundary = RectObs(np.array([[-22, 20],[22, 22.0]]), color=color)
    leftboundary = RectObs(np.array([[-22, -20],[-20, 20.0]]), color=color)
    rightboundary = RectObs(np.array([[20, -20],[22, 20.0]]), color=color)
    obs_list_list = []
    for i in range(case_num):
        print(i)

        obs_num = np.random.randint(obs_num_max-2,obs_num_max+1)
        obs_list = []
        ax.cla()
        for j in range(obs_num):
            while True:
                leftbottom = np.random.uniform(bounds[:, 0], bounds[:, 1]-5)
                width = np.random.uniform(*size_range)
                height = np.random.uniform(*size_range)
                left, bottom = leftbottom
                right, top = left+width, bottom+height
                if not 40 <= width*height <= 150: continue
                if right>bounds[0,1] or top>bounds[1,1]:
                    continue
                if j == 0:
                    break

                # the new obstacle should be min_gap away from others
                check = True
                for j_ in range(j):
                    
                    aabb1 = np.array([[left, bottom],[right,top]])
                    aabb2 = obs_list[j_].rect+np.array([[-min_gap],[min_gap]])

                    if aabb_intersect_aabb(aabb1, aabb2):
                        check = False
                if check:
                    break
            
            rect = np.array([leftbottom, leftbottom + np.array([width, height])])
            obs_list.append(RectObs(rect, color=color))
        obs_list += [botboundary, topboundary, leftboundary, rightboundary]
            
        plot_obs_list(ax, obs_list)
        obs_list_list.append(obs_list)
        
        plt.axis('equal')
        plt.axis([*bounds[0], *bounds[1]])
        plt.savefig(os.path.join(dir_path, 'obs'+str(i)+'.jpg'))

    pickle.dump(obs_list_list, open(os.path.join(dir_path,'obs_list_list.pkl'), 'wb'))