from crl_kino.env import DifferentialDriveGym
from tianshou.data import Collector, ReplayBuffer, Batch
import torch
from tianshou.policy import DDPGPolicy

from .net import Actor, Critic, WeightsActor
#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np

device = 'cpu'


def load_policy(env, layer, model_path):
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    action_range = [env.action_space.low, env.action_space.high]
    actor = Actor(
        layer, state_shape, action_shape,
        action_range, device
    ).to(device)
    critic = Critic(
        layer, state_shape, action_shape, device
    ).to(device)

    # actor critic
    actor = actor.to(device)
    actor_optim = torch.optim.Adam(actor.parameters())
    critic = critic.to(device)
    critic_optim = torch.optim.Adam(critic.parameters())
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        action_range=action_range)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    return policy


# def load_primitive_policy(env, layer, model_path):
#     state_shape = env.observation_space.shape or env.observation_space.n
#     action_shape = env.action_space.shape or env.action_space.n
#     action_range = [env.action_space.low, env.action_space.high]
#     actor = Actor(
#         layer, state_shape, action_shape,
#         action_range, device
#     ).to(device)
#     critic = Critic(
#         layer, state_shape, action_shape, device
#     ).to(device)
#     return load_policy(actor, critic, action_range, model_path)

# def load_composing_policy(env, layer, model_path):
#     state_shape = env.observation_space.shape or env.observation_space.n
#     action_shape = env.action_space.shape or env.action_space.n
#     action_range = [env.action_space.low, env.action_space.high]
#     actor = WeightsActor(
#         layer, state_shape, action_shape,
#         action_range, device
#     ).to(device)
#     critic = Critic(
#         layer, state_shape, action_shape, device
#     ).to(device)
#     return load_policy(actor, critic, action_range, model_path)


# def load_end2end_policy(env, layer, model_path):
#     state_shape = env.observation_space.shape or env.observation_space.n
#     action_shape = env.action_space.shape or env.action_space.n
#     action_range = [env.action_space.low, env.action_space.high]
#     actor = Actor(
#         layer, state_shape, action_shape,
#         action_range, device
#     ).to(device)
#     critic = Critic(
#         layer, state_shape, action_shape, device
#     ).to(device)
#     action_range = [env.action_space.low[0], env.action_space.high[0]]
#     return load_policy(actor, critic, action_range, model_path)

def policy_forward(policy, obs, info=None, eps=0.0):
    """
    Map the observation to the action under the policy,

    Parameters
    ----------
    policy: 
        a trained tianshou ddpg policy
    obs: array_like
        observation 
    info: 
        gym info
    eps: float
        The predicted action is extracted from an Gaussian distribution,
    eps*I is the covariance
    """
    obs = np.array(obs)
    obs_len = 1
    if obs.ndim == 1:
        obs_len = 1
    elif obs.ndim == 2:
        obs_len = len(obs)
    obs = obs.reshape((obs_len,-1))
    batch = Batch(obs=obs, info=None)
    batch = policy(batch, eps=eps)  
    act = batch.act.detach().cpu().numpy()
    return act