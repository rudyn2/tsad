import random
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sac.utils as utils

from agents.agent import MultiTaskActor, MultiTaskCritic
from agents.squashed_gaussian import SquashedGaussianMLP, mlp


class SACActor(MultiTaskActor, nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: tuple,
                 output_dim: int,
                 action_range: tuple):
        super().__init__()

        self._actors = nn.ModuleDict({
            str(hlc): SquashedGaussianMLP(
                input_dim,
                output_dim,
                hidden_dim,
                nn.ReLU
            ) for hlc in range(4)
        })
        self._action_range = action_range

    def act_batch(self, obs: torch.Tensor, task: int) -> Union[list, torch.Tensor]:
        pass

    def act_single(self, obs: torch.Tensor, task: int) -> list:
        pi_distribution = self._actors[str(task)].get_distribution(obs)
        if sample:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = pi_distribution.mean
        pi_action = torch.tanh(pi_action)
        return list(utils.to_np(pi_action.clamp(*self._action_range)))

    def forward(self, obs: torch.Tensor, hlc: int):
        """
        Returns actions and log-probability over those actions conditioned to the observations.
        For action selection please use act_batch or act_single.
        """
        pi_distribution = self._actors[str(hlc)].get_distribution(obs)
        pi_action = pi_distribution.rsample()
        pi_action = torch.tanh(pi_action)
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        return pi_action, logp_pi

    def get_supervised_loss(self, obs: torch.Tensor, act: torch.Tensor, task: int) -> torch.Tensor:
        pi_distribution = self._actors[str(task)].get_distribution(obs)
        pred_actions = pi_distribution.mean
        loss = F.mse_loss(pred_actions, act)
        return loss

    def train_mode(self):
        self._actors.train()

    def eval_mode(self):
        self._actors.eval()

    def save(self):
        pass

    def load(self):
        pass


class SACCritic(MultiTaskCritic, nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: tuple):
        super().__init__()

        # a critic just returns 1 value: the Q-value.
        self._critics1 = nn.ModuleDict({
            str(hlc): mlp([input_dim] + list(hidden_dim) + [1], nn.ReLU) for hlc in range(4)
        })

        self._critics2 = nn.ModuleDict({
            str(hlc): mlp([input_dim] + list(hidden_dim) + [1], nn.ReLU) for hlc in range(4)
        })

    def forward(self, obs: torch.Tensor, act: torch.Tensor, hlc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        output_dim: (batch_size,)
        """
        assert obs.size(0) == act.size(0)
        obs_action = torch.cat([obs, act], dim=1)
        q1 = self._critics1[str(hlc)](obs_action).squeeze(-1)
        q2 = self._critics2[str(hlc)](obs_action).squeeze(-1)
        return q1, q2

    def train_mode(self):
        self._critics1.train()
        self._critics2.train()

    def eval_mode(self):
        self._critics1.eval()
        self._critics2.eval()

    def save(self):
        pass

    def load(self):
        pass


if __name__ == "__main__":
    batch_size = 128
    obs_dim = 15
    act_dim = 2
    act_range = (-1, 1)
    hlc = random.choice([0, 1, 2, 3])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create fake data
    obs = torch.rand((batch_size, obs_dim), requires_grad=True, device=device)
    act = torch.rand((batch_size, act_dim), requires_grad=True, device=device) * act_range[1]

    # create sac actor and critic
    critic = SACCritic(input_dim=obs_dim, hidden_dim=(128, 64))
    actor = SACActor(input_dim=obs_dim, hidden_dim=(128, 64), output_dim=act_dim)
    critic.to(device=device)
    actor.to(device=device)

    # test them
    q1, q2 = critic(obs, act, hlc)
    assert q1.size(0) == batch_size and len(q1.size()) == 1
    assert q2.size(0) == batch_size and len(q2.size()) == 1

    action, logp = actor(obs, hlc)
    assert action.size(0) == batch_size and action.size(1) == act_dim
    assert logp.size(0) == batch_size and len(logp.size()) == 1
    assert torch.sum(logp > 0) == 0


