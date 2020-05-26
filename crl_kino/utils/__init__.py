import os.path, os
import pickle
from crl_kino.utils.draw import *


def obs_generate(case_num=10, obs_num=7, size=2.5, min_gap=3.0, bounds=np.array([[-20.0, 20.0], [-20.0, 20.0]]), dir_path='.'):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    fig, ax = plt.subplots()
    obc_list = np.zeros((case_num, obs_num, 2))
    for i in range(case_num):
        for j in range(obs_num):
            while True:
                obc = np.random.uniform(bounds[:, 0], bounds[:, 1])
                if j == 0:
                    break
                # check gap
                gap = np.min(np.linalg.norm(
                    obc_list[i, :j, :]-obc, axis=1))-2*size
                if gap >= min_gap:
                    break
            obc_list[i, j] = obc
        ax.cla()
        plot_ob(ax, obc_list[i, :, :], obs_size=size)
        # plt.xlim(bounds[0])

        plt.axis('equal')
        plt.ylim(bounds[1])
        plt.savefig(os.path.join(dir_path, 'obs'+str(i)+'.jpg'))
    pickle.dump(obc_list, open(os.path.join(dir_path,'obs.pkl'), 'wb'))


__all__ = [
    'obs_generate',
]