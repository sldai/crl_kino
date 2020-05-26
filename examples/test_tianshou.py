#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
from matplotlib import pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from differential_gym import DifferentialDriveGym

import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import VectorEnv
from tianshou.policy import DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, Batch
from train_tianshou import get_args
from net import Actor, Critic
import pickle



def collect_data(env, policy, ret_images = False):
    for _ in range(3):
        obs = env.reset()
    obs = env._obs()
    images = []
    while True:
        if ret_images:
            env.render(pause=False)
            canvas = FigureCanvas(plt.gcf())
            canvas.draw()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
            images.append(image)
        obs_batch = Batch(obs=obs.reshape((1, -1)), info=None)
        action_batch = policy.forward(obs_batch, deterministic=True)
        action = action_batch.act
        action = action.detach().numpy().flatten()
        # print(action)

        obs, rewards, done, info = env.step(action)
        if done:
            break
    return info, env.current_time, images


def main(args = get_args()):
    torch.set_num_threads(1)  # we just need only one thread for NN
    env = DifferentialDriveGym()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    actor = Actor(
        args.layer, args.state_shape, args.action_shape,
        args.max_action, args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic = Critic(
        args.layer, args.state_shape, args.action_shape, args.device
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        args.tau, args.gamma, args.exploration_noise,
        [env.action_space.low[0], env.action_space.high[0]],
        reward_normalization=args.rew_norm, ignore_done=False)
    log_path = os.path.join(args.logdir, args.task, 'ddpg')
    model_path = os.path.join(log_path, 'policy.pth')
    if not os.path.exists(model_path): print('Model does not exist')
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))

    # run model in train and test environment
    run_num = 200
    obc_list = pickle.load(open('obc_list.pkl', 'rb'))
    obc_list_test = pickle.load(open('obc_list_test.pkl', 'rb'))
    train_env = DifferentialDriveGym()
    train_env.obc_list = obc_list[:20]
    test_env = DifferentialDriveGym()
    test_env.obc_list = obc_list_test

    suc_rate_train = np.zeros(run_num)
    time_train = np.zeros(run_num)
    for i in range(1):
        info, ctime, images = collect_data(train_env, policy, ret_images=True)
        time_train[i] = ctime
        imageio.mimsave(os.path.join('gif', 'train_collsion_'+str(i)+'.gif'), images, fps=10)
        if info['goal']:
            suc_rate_train[i] = 1
            
        elif info['collision']:
            suc_rate_train[i] = 0
            # imageio.mimsave(os.path.join('gif', 'train_collsion_'+str(i)+'.gif'), images, fps=10)
        else:
            suc_rate_train[i] = 0
            # imageio.mimsave(os.path.join('gif', 'train_not_reach_'+str(i)+'.gif'), images, fps=10)
        print(i)
    suc_rate_train = np.sum(suc_rate_train)/np.size(suc_rate_train)

    return
    suc_rate_test = np.zeros(run_num)
    time_test = np.zeros(run_num)
    for i in range(run_num):
        info, ctime, images = collect_data(test_env, policy)
        time_test[i] = ctime
        if info['goal']:
            suc_rate_test[i] = 1
            
        elif info['collision']:
            suc_rate_test[i] = 0
            # imageio.mimsave(os.path.join('gif', 'test_collsion_'+str(i)), images, fps=10)
        else:
            suc_rate_test[i] = 0
            # imageio.mimsave(os.path.join('gif', 'test_not_reach_'+str(i)), images, fps=10)
        print(i)
    suc_rate_test = np.sum(suc_rate_test)/np.size(suc_rate_test)

    print(suc_rate_train, suc_rate_test)
    np.savetxt('time.csv',np.array([time_train, time_test]),delimiter=',')
    plt.hist(time_train, label='train', cumulative=True, histtype='step')
    plt.hist(time_test, label='test', cumulative=True, histtype='step')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()