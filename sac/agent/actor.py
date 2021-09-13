from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

import sac.utils as utils
from models.ActorCritic import Actor

LOG_STD_MAX = 2
LOG_STD_MIN = -20
epsilon = 1e-6
ActionSpace = namedtuple("ActionSpace", "high low")


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, input_size, action_dim, hidden_dim, log_std_bounds,
                 action_space=ActionSpace(high=np.array([1, 1]), low=np.array([0, -1]))):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.actor = Actor(input_size, hidden_dim, action_dim)
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.apply(utils.weight_init)
        self.action_scale = self.action_scale.to(self._device)
        self.action_bias = self.action_bias.to(self._device)

    def forward(self, obs, hlc):
        mu, log_std = self.actor(obs, hlc).chunk(2, dim=-1)
        return None
