from crl_kino.utils import obs_generate
import os.path
from crl_kino.env import DifferentialDriveEnv, DifferentialDriveGym
from crl_kino.policy.dwa import DWA
from crl_kino.utils.draw import *
from crl_kino.planner.rrt import RRT
from crl_kino.planner.rrt_rl import RRT_RL
from crl_kino.policy.rl_policy import load_RL_policy
from crl_kino.planner.sst import SST
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='rl_rrt', 
                        help='test different cases: ' +
                        '<--case env> test simulation class DifferentialDriveEnv'+
                        '<--case gym> test gym env DifferentialDriveGym'+
                        '<--case rrt> test kino rrt with dwa steering'+
                        '<--case rl_rrt> test kino rrt with deep RL')
    args = parser.parse_known_args()[0]
    return args

def test_env():
    env = DifferentialDriveEnv(1.0, -0.1, np.pi, 1.0, np.pi)
    dwa = DWA(env)
    dwa.set_dwa(dt=0.2, to_goal_cost_gain=1.2)
    obs = np.array([[-10.402568,   -5.5128484],
                    [14.448388,   -4.1362205],
                    [10.003768,   -1.2370133],
                    [11.609167,    0.9119211],
                    [-4.9821305,   3.8099794],
                    [8.94005,    -4.14619],
                    [-10.45487,     6.000557]])
    env.set_obs(obs)
    start = np.array([13, -7.5, 0, 0, 0.0])
    goal = np.array([10, 10, 0, 0, 0.0])
    state = start.copy()

    fig, ax = plt.subplots()
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
        plt.cla()
        plot_ob(ax, env.obs_list, env.obs_size)
        plt.plot(traj[:, 0], traj[:, 1], "-c")
        plt.plot(state[0], state[1], "xr")
        plt.plot(goal[0], goal[1], "xb")
        plot_arrow(state[0], state[1], state[2])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
    plt.show()


def test_gym():
    '''
    debug gym
    '''
    env = DifferentialDriveGym()
    env.set_curriculum(ori=False, obs_num=7)
    env.reset()

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
        obs, reward, done, info = env.step(env.v2a(v))
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

    policy = load_RL_policy([1024, 768, 512], os.path.join(
        os.path.dirname(os.path.dirname(__file__))
        , 'data/log/mid_noise/ddpg/policy.pth'))
    planner = RRT_RL(env, policy)
    start = np.array([13, -7.5, 0, 0, 0.0])
    goal = np.array([10, 10, 0, 0, 0.0])

    planner.set_start_and_goal(start, goal)
    path = planner.planning()
    planner.draw_path(path)
    planner.draw_tree()


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
    find_exact_solution = sst.planning()
    print(find_exact_solution)
    fig, ax = plt.subplots()
    plot_problem_definition(ax, sst.robot_env.obs_list,
                            sst.robot_env.obs_size, sst.robot_env.robot_radius,
                            sst.obRealVector2array(sst.start), sst.obRealVector2array(sst.goal))

    
    planner_data = sst.planner_data
    for edge in planner_data['edges']:
        pair = planner_data['nodes'][edge]
        plt.plot(pair[:,0], pair[:,1], '-r', linewidth=0.6)

    plt.plot(sst.path[:, 0], sst.path[:, 1], '-b', linewidth=2.0)
    plt.show()

if __name__ == "__main__":
    args = get_args()
    print(args)
    if args.case == 'env': test_env()
    elif args.case == 'gym': test_gym()
    elif args.case == 'rrt': test_rrt()
    elif args.case == 'rl_rrt': test_rl_rrt()
    elif args.case == 'sst': test_sst()
