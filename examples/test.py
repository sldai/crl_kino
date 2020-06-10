from crl_kino.policy.rl_policy import load_policy, policy_forward
from crl_kino.utils import obs_generate
import os.path
from crl_kino.env import DifferentialDriveEnv, DifferentialDriveGym
from crl_kino.policy.dwa import DWA
from crl_kino.utils.draw import *
from crl_kino.planner.rrt import RRT
from crl_kino.planner.rrt_rl import RRT_RL

try:
    from crl_kino.planner.sst import SST
except ImportError:
    print('ompl not installed, cannot use SST')
import argparse
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='rl_rrt',
                        help='test different cases: ' +
                        '<--case env> test simulation class DifferentialDriveEnv' +
                        '<--case gym> test gym env DifferentialDriveGym' +
                        '<--case rrt> test kino rrt with dwa steering' +
                        '<--case rl_rrt> test kino rrt with deep RL')
    args = parser.parse_known_args()[0]
    return args


def test_env():
    env = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)
    dwa = DWA(env)
    dwa.set_dwa(dt=0.2, to_goal_cost_gain=1.2)
    obs_list = pickle.load(open(os.path.dirname(__file__)+'/data/obstacles/obs_list_list.pkl', 'rb'))[0]

    env.set_obs(obs_list)
    start = np.array([-10, -15.0, 0, 0, 0.0])
    goal = np.array([0, -15, 0, 0, 0.0])
    state = start.copy()

    # fig, ax = plt.subplots()
    path = [state.copy()]
    for i in range(200):
        v, traj = dwa.control(state, goal)

        print(v)
        state = env.motion_velocity(state, v, 1.0/5.0)

        if not env.valid_state_check(state):
            print('Collision')
            break
        if np.linalg.norm(state[:2]-goal[:2]) < .6:
            print('Goal')
            break
        # plt.cla()
        # plot_problem_definition(ax, env.obs_list, env.rigid_robot, start, goal)
        # plt.plot(traj[:, 0], traj[:, 1], "-c")
        # plot_robot(ax, env.rigid_robot, state[:3])
        # plt.axis("equal")
        # plt.grid(True)
        # plt.pause(0.0001)
        path.append(state.copy())
    draw_path(env, start, goal, path)
    plt.show()


def test_gym():
    '''
    debug gym
    '''
    obs_list = pickle.load(open(os.path.dirname(__file__)+'/data/obstacles/obs_list_list.pkl', 'rb'))[:1]
    # start = np.array([-10, -15.0, 0, 0, 0.0])
    # goal = np.array([0, -15, 0, 0, 0.0])

    env = DifferentialDriveGym(obs_list_list=obs_list)
    env.reset()
    # env.state = start
    # env.goal = goal

    dwa = DWA(env.robot_env)
    dwa.set_dwa(dt=0.2)

    

    start = env.state
    state = start.copy()
    goal = env.goal

    fig, ax = plt.subplots()
    rew = 0.0
    for i in range(200):
        v, traj = dwa.control(state, goal)
        print(v)
        obs, reward, done, info = env.step(v)
        rew += reward
        print(reward)
        # print(info)
        # print(obs[:4])
        state = env.state
        if done:
            print(done, rew)
        if info['collision']:
            print('Collision')
            break
        if info['goal']:
            print('Goal')
            break

        env.render(plot_localwindow=True)
    plt.show()


def test_policy():
    # torch.set_num_threads(1)  # we just need only one thread for NN
    model_path =os.path.dirname(__file__)+'/../data/net/end2end/ddpg/policy.pth'
  
    env = DifferentialDriveGym()
    obstacles = np.array([[-10.402568,   -5.5128484],
                          [14.448388,   -4.1362205],
                          [10.003768,   -1.2370133],
                          [11.609167,    0.9119211],
                          [-4.9821305,   3.8099794],
                          [8.94005,    -4.14619],
                          [-10.45487,     6.000557]])

    env.obc_list = [obstacles]
    policy = load_policy(env, [512, 512, 512], model_path)

    obs = env.reset()
    env.robot_env.set_obs(obstacles)

    while True:
        action = policy_forward(policy, obs, eps=0.05)
        print(action)
        obs, reward, done, info = env.step(action[0])
        env.render()
        if done:
            break


def test_rrt():
    env = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)
    obs = np.array([[-10.402568,   -5.5128484],
                    [14.448388,   -4.1362205],
                    [10.003768,   -1.2370133],
                    [11.609167,    0.9119211],
                    [-4.9821305,   3.8099794],
                    [8.94005,    -4.14619],
                    [-10.45487,     6.000557]])
    env.set_obs(obs)
    planner = RRT(env)

    start = np.array([13, -7.5, 0, 0, 0.0])
    goal = np.array([10, 10, 0, 0, 0.0])

    planner.set_start_and_goal(start, goal)
    path = planner.planning()
    planner.draw_path(path)
    planner.draw_tree()


def test_rl_rrt():
    env = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)
    obs = np.array([[-10.402568,   -5.5128484],
                    [14.448388,   -4.1362205],
                    [10.003768,   -1.2370133],
                    [11.609167,    0.9119211],
                    [-4.9821305,   3.8099794],
                    [8.94005,    -4.14619],
                    [-10.45487,     6.000557]])
    env.set_obs(obs)
    model_path =os.path.dirname(__file__)+'/../data/net/end2end/ddpg/policy.pth'
    policy = load_policy(DifferentialDriveGym(), [512,512,512], model_path)
    planner = RRT_RL(env, policy)
    start = np.array([13, -7.5, 0, 0, 0.0])
    goal = np.array([10, 10, 0, 0, 0.0])

    planner.set_start_and_goal(start, goal)
    path = planner.planning()
    draw_path(env, start, goal, path)
    draw_tree(env, start, goal, planner.node_list)
    # planner.draw_tree()


def test_sst():
    env = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)
    obs_list = pickle.load(open(os.path.dirname(__file__)+'/data/obstacles/obs_list_list.pkl', 'rb'))[0]

    env.set_obs(obs_list)

    sst = SST(env)
    start = np.array([-5, -15, 0, 0, 0.0])
    goal = np.array([-15, 15, 0, 0, 0.0])

    sst.set_start_and_goal(start, goal)
    find_exact_solution = sst.planning(2000)

    fig, ax = plt.subplots()
    plot_problem_definition(ax, sst.robot_env.obs_list, sst.robot_env.rigid_robot,
                            sst.obRealVector2array(sst.start), sst.obRealVector2array(sst.goal))

    planner_data = sst.planner_data
    for edge in planner_data['edges']:
        pair = planner_data['nodes'][edge]
        plt.plot(pair[:, 0], pair[:, 1], '-r', linewidth=0.6)

    plt.plot(sst.path[:, 0], sst.path[:, 1], '-b', linewidth=2.0)
    plt.savefig('sst.png')

    draw_path(sst.robot_env, start, goal, sst.path, fname='sst_path')
    plt.show()




if __name__ == "__main__":
    args = get_args()
    print(args)
    if args.case == 'env':
        test_env()
    elif args.case == 'gym':
        test_gym()
    elif args.case == 'policy':
        test_policy()
    elif args.case == 'rrt':
        test_rrt()
    elif args.case == 'rl_rrt':
        test_rl_rrt()
    elif args.case == 'sst':
        test_sst()
