from crl_kino.planner.rrt import RRT
import time
import numpy as np

class RRTStar(RRT):
    def planning(self):
        """
        rrt path planning
        """
        if self.start is None or self.goal is None:
            raise ValueError('start or goal is not set')
        tic = time.time()
        self.node_list = [self.start]
        path = None
        reach_exactly = False

        best_node = None
        for i in range(self.max_iter):
            print('Iteration {}: {} nodes'.format(i, len(self.node_list)))
            good_sample = False
            while not good_sample:
                rnd_node = self.sample(self.goal)
                if not self.robot_env.valid_state_check(rnd_node.state): 
                    continue
                nearest_node, cost = self.nearest(self.node_list, rnd_node, self.Eu_metric)
                if cost < self.epsilon: 
                    good_sample = True
            
            parent_node = self.choose_parent(rnd_node)
            new_node_list = self.steer(parent_node, rnd_node)

            if len(new_node_list)>0:
                near_nodes = self.near(new_node_list, self.goal, self.se2_metric, 1.5)
                if len(near_nodes)>0:
                    reach_exactly = True
                    near_node = min(near_nodes, key=lambda node: node.cost)
                    if best_node == None or near_node.cost<best_node.cost:
                        best_node = near_node
                    print('Path length: {}'.format(best_node.cost))
        if best_node == None:
            best_node, dis = self.self.nearest(self.node_list, self.goal, self.se2_metric)
        path = self.generate_final_course(best_node)
        toc = time.time()
        self.planning_time = toc-tic
        self.path = path
        self.reach_exactly = reach_exactly
        return path
        

    def choose_parent(self, rnd_node):
        """
        Choose the node to extend after sampling a random node,
        the chosen node has the lowest cost, i.e. h=g+c 
        """
        near_nodes = self.near(self.node_list, rnd_node, self.Eu_metric, self.epsilon)
        parent_node = None
        if near_nodes == None:
            parent_node, min_c_cost = self.nearest(self.node_list, rnd_node, self.Eu_metric)
        else:
            costs = np.array([node.cost + self.ttr_metric(node, rnd_node) for node in near_nodes])
            min_ind = np.argmin(costs)
            parent_node, min_h_cost = near_nodes[min_ind], costs[min_ind]
        return parent_node

if __name__ == "__main__":
    import pickle, os
    from crl_kino.estimator.network import TTRCU
    import torch
    from crl_kino.env.dubin_env import DubinEnv
    from crl_kino.env.dubin_gym import DubinGym, DubinGymCU
    from crl_kino.policy.rl_policy import policy_forward, load_policy
    from crl_kino.utils.draw import *
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

    # planner = RRTStar(env, policy=policy, ttr_estimator=estimator, max_iter=1000)

    # planner.set_start_and_goal(start, goal)

    # path = planner.planning()

    planner = pickle.load(open('rrt.pkl', 'rb'))
    draw_path(env, start, goal, planner.path, fname='rrt_path')
    draw_tree(env, start, goal, planner.node_list, fname='rrt_tree')
    # pickle.dump(planner, open('rrtstar.pkl', 'wb'))