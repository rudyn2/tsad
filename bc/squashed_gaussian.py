import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple, activation: nn.Module, act_limit: int = 1):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action

    def logp_pi(self, obs, act):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        logp_pi = pi_distribution.log_prob(act).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(axis=1)
        return logp_pi