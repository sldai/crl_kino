import torch
from tianshou.policy import DDPGPolicy

from .net import Actor, Critic
from tianshou.data import Collector, ReplayBuffer, Batch
from crl_kino.env import DifferentialDriveGym

def load_RL_policy(layer, model_path):
    device = 'cpu'
    torch.set_num_threads(1)  # we just need only one thread for NN
    env = DifferentialDriveGym()
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # model
    actor = Actor(
        layer, state_shape, action_shape,
        max_action, device
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters())
    critic = Critic(
        layer, state_shape, action_shape, device
    ).to(device)
    critic_optim = torch.optim.Adam(critic.parameters())
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        action_range=[env.action_space.low[0], env.action_space.high[0]])
    policy.load_state_dict(torch.load(model_path, map_location=device))
    return policy