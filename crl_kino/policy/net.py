import torch
import numpy as np
from torch import nn


class Actor(nn.Module):
    def __init__(self, layer, state_shape, action_shape,
                 max_action, device='cpu'):
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
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        logits = self._max * torch.tanh(logits)
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