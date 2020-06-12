import torch
import numpy as np
from torch import nn

class WeightsActor(nn.Module):
    def __init__(self, layer, state_shape, action_shape,
                 action_range, device='cpu'):
        """
        Parameters
        ----------
        list layer: [int, int, ...] 
            network hidden layer
        """

        super().__init__()
        self.device = device
        
        self.model = [
            nn.Linear(np.prod(state_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(layer[-1], np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self.low = torch.tensor(action_range[0], device=self.device)
        self.high = torch.tensor(action_range[1], device=self.device)

        self.action_bias = (self.low+self.high)/2
        self.action_scale = (self.high-self.low)/2

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        logits = torch.softmax(logits, dim=1)
        # noise
        if kwargs.get('eps') is not None:
            eps = kwargs['eps']
            noise = torch.randn(size=logits.shape, device=logits.device) * eps
            noise -= torch.mean(noise, dim=1).view((-1,1))
            logits = logits + noise
        # clip
        logits = torch.min(torch.max(logits, self.low.view((1,-1))), self.high.view((1,-1)))
        logits = logits / torch.sum(logits, dim=1).view((-1,1))
        return logits, 'softmax'

class Conv(nn.Module):
    def __init__(self, input_size = [1, 30, 30], output_size = 100, device='cpu'):
        super.__init__()
        self.device = device
        self.model = [
            nn.Conv2d(1,8,kernel_size=(5,5)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.PReLU(),
            nn.Conv2d(8,16,kernel_size=(3,3)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.PReLU(),
            nn.Conv2d(8,16,kernel_size=(3,3)),
            nn.PReLU()
        ]
        self.model = nn.Sequential(*self.model)
        rand_sample = torch.rand([1]+input_size)
        length = self.model(rand_sample).flatten().size()
        self.model.add_module('flatten layer',nn.Linear(length, output_size))

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        logits = self.model(s)
        return logits

class ActorConv(nn.Module):
    """
    The actor and critic share a convolutional network, which encodes the local map
    into a vector. Then the actor and critic combines the 
    """
    def __init__(self, conv, actor):
        self.conv = conv
        self.actor = actor
        self.device = self.actor.device
    
    def forward(self, tuple_state, **kwargs):
        local_map, s = tuple_state
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        logits = self.conv(local_map)
        logits = torch.cat(logits,s)
        logits = self.actor(logits,**kwargs)
        return logits

class CriticConv(nn.Module):
    """
    The actor and critic share a convolutional network, which encodes the local map
    into a vector. Then the actor and critic combines the 
    """
    def __init__(self, conv, critic):
        self.conv = conv
        self.critic = critic
        self.device = self.critic.device
    
    def forward(self, tuple_state, **kwargs):
        local_map, s = tuple_state
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        logits = self.conv(local_map)
        logits = torch.cat(logits,s)
        logits = self.critic(logits,**kwargs)
        return logits



class Actor(nn.Module):
    def __init__(self, layer, state_shape, action_shape,
                 action_range, device='cpu'):
        """
        Parameters
        ----------
        list layer: [int, int, ...] 
            network hidden layer
        """

        super().__init__()
        self.device = device
        
        self.model = [
            nn.Linear(np.prod(state_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(layer[-1], np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self.low = torch.tensor(action_range[0], device=self.device)
        self.high = torch.tensor(action_range[1], device=self.device)

        self.action_bias = (self.low+self.high)/2
        self.action_scale = (self.high-self.low)/2

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        logits = torch.tanh(logits)
        if kwargs.get('eps') is not None:
            eps = kwargs['eps']
            logits = logits + torch.randn(
                    size=logits.shape, device=logits.device) * eps
        # scale the logits to produce actions
        logits = logits * self.action_scale.view((1,-1)) + self.action_bias.view((1,-1))
        logits = torch.min(torch.max(logits, self.low.view((1,-1))), self.high.view((1,-1)))
        return logits, None


class ActorProb(nn.Module):
    def __init__(self, layer, state_shape, action_shape,
                 max_action, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(layer[-1], np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class Critic(nn.Module):
    def __init__(self, layer, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), layer[0]),
            nn.ReLU(inplace=True)]
        for i in range(len(layer)-1):
            self.model += [nn.Linear(layer[i], layer[i+1]), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(layer[-1], 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits