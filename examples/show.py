import pickle, os
from crl_kino.estimator.network import TTRCU
import torch
from crl_kino.env.dubin_env import DubinEnv
from crl_kino.env.dubin_gym import DubinGym, DubinGymCU
from crl_kino.policy.rl_policy import policy_forward, load_policy
from crl_kino.utils.draw import *
from crl_kino.planner.rrt import RRT
from crl_kino.planner.rrt_star import RRTStar
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='rrtstar')
    args = parser.parse_known_args()[0]
    return args


def rrt():
    # test_env1 = pickle.load(open('data/obstacles/test_env1.pkl', 'rb'))
    # start = np.array([-5, -15, 0, 0, 0.0])
    # goal = np.array([10, 10, 0, 0, 0.0])

    test_env2 = pickle.load(open('data/obstacles/test_env2.pkl', 'rb'))
    start = np.array([-15.0,17,0])
    goal = np.array([10.8,-8.5,0])

    # robot_env
    env, gym = DubinEnv(), DubinGymCU()
    env.set_obs(test_env2)

    # policy
    model_path = os.path.join('log', 'dubin', 'ddpg/policy.pth')
    policy = load_policy(gym, [1024, 512, 512, 512], model_path)

    # ttr_estimator
    estimator = TTRCU(gym.observation_space.shape[0], 1, 'cpu').to('cpu')
    estimator.load_state_dict(torch.load('log/estimator/estimator.pth', map_location='cpu'))

    planner = RRT(env, policy=policy, ttr_estimator=estimator, max_iter=1000)

    planner.set_start_and_goal(start, goal)

    path = planner.planning()
    draw_path(env, start, goal, path)
    draw_tree(env, start, goal, planner.node_list)
    pickle.dump(planner, open('rrt.pkl','wb'))

def rrtstar():

    # test_env1 = pickle.load(open('data/obstacles/test_env1.pkl', 'rb'))
    # start = np.array([-5, -15, 0, 0, 0.0])
    # goal = np.array([10, 10, 0, 0, 0.0])

    test_env2 = pickle.load(open('data/obstacles/test_env2.pkl', 'rb'))
    start = np.array([-15.0,17,0])
    goal = np.array([10.8,-8.5,0])

    # robot_env
    env, gym = DubinEnv(), DubinGymCU()
    env.set_obs(test_env2)

    # policy
    model_path = os.path.join('log', 'dubin', 'ddpg/policy.pth')
    policy = load_policy(gym, [1024, 512, 512, 512], model_path)

    # ttr_estimator
    estimator = TTRCU(gym.observation_space.shape[0], 1, 'cpu').to('cpu')
    estimator.load_state_dict(torch.load('log/estimator/estimator.pth', map_location='cpu'))

    planner = RRTStar(env, policy=policy, ttr_estimator=estimator, max_iter=1000)
    planner.set_start_and_goal(start, goal)
    path = planner.planning()

    draw_path(env, start, goal, planner.path, fname='rrt_path')
    draw_tree(env, start, goal, planner.node_list, fname='rrt_tree')
    # pickle.dump(planner, open('rrtstar.pkl', 'wb'))

if __name__ == "__main__":
    args = get_args()
    if args.case == 'rrt':
        rrt()
    elif args.case == 'rrtstar':
        rrtstar()
    