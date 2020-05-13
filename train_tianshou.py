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

from net import Actor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True, help='train or test')
    parser.add_argument('--task', type=str, default='DifferentialDrive-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer_size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=2400)
    parser.add_argument('--collect-per-step', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer', type=list, default=[1024, 768, 512])
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=bool, default=True)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--rew_param', type=str, default='normal')
    args = parser.parse_known_args()[0]
    return args


def reward_param(name):
    if name == 'normal':
        return np.array([300.0, -0.48, -1.0, -200.0, 1.0, 1.0, -1.0, -1.0])
    elif name == 'less_heading':
        return np.array([300.0, -0.48, -0.5, -200.0, 1.0, 1.0, -1.0, -1.0])
    elif name == 'more_step':
        return np.array([300.0, -0.48, -1.0, -200.0, 1.0, 1.0, -1.0, -2.0])


def test_ddpg(args=get_args()):
    torch.set_num_threads(1)  # we just need only one thread for NN
    env = DifferentialDriveGym(reward_param=reward_param(args.rew_param))
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    train_envs = VectorEnv(
        [lambda: DifferentialDriveGym(reward_param=reward_param(args.rew_param)) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = VectorEnv(
        [lambda: DifferentialDriveGym(reward_param=reward_param(args.rew_param)) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
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
        reward_normalization=args.rew_norm, ignore_done=True)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ddpg')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= 100

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, save_fn=save_fn, writer=writer)
    train_collector.close()
    test_collector.close()
    # if __name__ == '__main__':
    #     pprint.pprint(result)
    #     # Let's watch its performance!
    #     env = DifferentialDriveGym()
    #     collector = Collector(policy, env)
    #     result = collector.collect(n_episode=1, render=args.render)
    #     print(f'Final reward: {result["rew"]}, length: {result["len"]}')
    #     collector.close()


def test_trained(args=get_args()):
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
    policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth')))
    if __name__ == '__main__':
        # Let's watch its performance!
        env = DifferentialDriveGym()

        # obs = env.reset()

        # env.state[0] = 4.0
        # env.state[1] = -18.0
        # env.state[2] = 0.0
        # env.goal[0] = 10.0
        # env.goal[1] = -10

        # obs = env._obs()

        # images = []
        # while True:
        #     env.render(pause=False)
        #     canvas = FigureCanvas(plt.gcf())
        #     canvas.draw()
        #     image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        #     image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        #     images.append(image)
        #     obs_batch = Batch(obs=obs.reshape((1, -1)), info=None)
        #     action_batch = policy.forward(obs_batch, deterministic=True)
        #     action = action_batch.act
        #     action = action.detach().numpy().flatten()
        #     print(action)

        #     obs, rewards, done, info = env.step(action)
        #     if done:
        #         break
        # imageio.mimsave('collision_avoid.gif', images, fps=5)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=2, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    args = get_args()
    if args.train:
        test_ddpg(args)
    else:
        test_trained(args)
