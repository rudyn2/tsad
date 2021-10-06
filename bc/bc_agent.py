from typing import Union

import numpy as np
import torch.cuda
import torch.nn as nn
from torch.optim.adam import Adam

from models.ActorCritic import Actor
from agents.squashed_gaussian import SquashedGaussianMLP
from agents.agent import MultiTaskActor
from bc.utils import unnormalize_pid_action_torch


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class BCStochasticAgent(MultiTaskActor):

    def __init__(self, **kwargs):
        self._actor = nn.ModuleDict({
            str(hlc): SquashedGaussianMLP(
                kwargs["input_size"],
                kwargs["action_dim"],
                (kwargs["hidden_dim"], kwargs["hidden_dim"]),
                nn.ReLU
            ) for hlc in range(4)
        })
        self._actor_optimizer = Adam(self._actor.parameters(), lr=0.0001)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._actor.to(self._device)
        self._mse = nn.MSELoss()
        self._checkpoint = kwargs['checkpoint']

    def act_batch(self, obs: list, task: int) -> torch.Tensor:
        with torch.no_grad():
            encoding = torch.stack([torch.tensor(o['encoding'], device=self._device) for o in obs], dim=0).float()
            action, _ = self._actor[str(task)].get_distribution(encoding)
            return action

    def act_single(self, obs: dict, task: int) -> list:
        encodings = torch.tensor(obs['affordances'], device=self._device).unsqueeze(0).float()
        with torch.no_grad():
            action, _ = self._actor[str(task)].get_distribution(encodings)  # (1, 2)
        return list(action.squeeze(dim=0).cpu().numpy())                    # [target_speed, steer]

    def get_supervised_loss(self, obs: torch.Tensor, act: torch.Tensor, task: int) -> float:
        pred_act, _ = self._actor[str(task)].get_distribution(obs)
        pred_act = unnormalize_pid_action_torch(pred_act)
        loss = self._mse(pred_act, act)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def train_mode(self):
        self._actor.train()

    def eval_mode(self):
        self._actor.eval()
    
    def save(self):
        torch.save(self._actor, self._checkpoint)
    
    def load(self):
        self._actor = torch.load(self._checkpoint)


class BCDeterministicAgent(MultiTaskActor):
    def load(self):
        pass

    def save(self):
        pass

    def __init__(self, **kwargs):
        self._actor = Actor(kwargs["input_size"], kwargs["hidden_dim"], kwargs["action_dim"], output_factor=1)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._actor.to(self._device)
        self._actor_optimizer = Adam(self._actor.parameters(), lr=0.00001)
        self._actor_loss = nn.MSELoss()

    def act_batch(self, obs: list, task: int) -> Union[list, torch.Tensor]:
        with torch.no_grad():
            pred_act = self._actor(obs, task)
            return pred_act

    def act_single(self, obs: dict, task: int) -> list:
        with torch.no_grad():
            pred_act = self._actor(obs, task)
            return pred_act

    def get_supervised_loss(self, obs: torch.Tensor, act: torch.Tensor, task: int) -> float:
        pred_act = self._actor(obs, task)
        mse_loss = self._actor_loss(pred_act, act)
        self._actor_optimizer.zero_grad()
        mse_loss.backward()
        self._actor_optimizer.step()
        return mse_loss.item()

    def train_mode(self):
        self._actor.train()

    def eval_mode(self):
        self._actor.eval()
