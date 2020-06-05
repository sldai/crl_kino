from crl_kino.env import DifferentialDriveGym
from tianshou.data import Collector, ReplayBuffer, Batch
import torch
from tianshou.policy import DDPGPolicy

from .net import Actor, Critic

device = 'cpu'


def load_policy(actor, critic, action_range, model_path):

    # model
    actor = actor.to(device)
    actor_optim = torch.optim.Adam(actor.parameters())
    critic = critic.to(device)
    critic_optim = torch.optim.Adam(critic.parameters())
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        action_range=action_range)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    return policy


def load_primitive_policy(env, model_path):
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    actor = Actor(
        [128], state_shape, action_shape,
        max_action, device
    ).to(device)
    critic = Critic(
        [128], state_shape, action_shape, device
    ).to(device)
    action_range = [env.action_space.low[0], env.action_space.high[0]]
    return load_policy(actor, critic, action_range, model_path)


def load_end2end_policy(env, model_path):
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    actor = Actor(
        [1024, 768, 512], state_shape, action_shape,
        max_action, device
    ).to(device)
    critic = Critic(
        [1024, 768, 512], state_shape, action_shape, device
    ).to(device)
    action_range = [env.action_space.low[0], env.action_space.high[0]]
    return load_policy(actor, critic, action_range, model_path)
