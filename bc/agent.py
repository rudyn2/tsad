from abc import abstractmethod
from typing import Union

import numpy as np
import torch.cuda
import torch.nn as nn
from torch.optim.adam import Adam

from sac.agent.actor import DiagGaussianActor
from models.ActorCritic import Actor


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class MultiTaskAgent:

    @abstractmethod
    def act_batch(self, obs: list, task: int) -> Union[list, torch.Tensor]:
        """
        Given a list of observations in the context of some task, return the predicted action for each observation.
        (the observation's type should be handled in the inherited classes)
        """
        raise NotImplementedError

    @abstractmethod
    def act_single(self, obs: dict, task: int) -> list:
        """
        Given a single observation in the context of some task, return the predicted action.
        (the observation's type should be handled in the inherited classes)
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, obs: list, act: list, task: int) -> float:
        """
        Given a list of observations in the context of some task, update the agent and return some metric (e.g. loss).
        (the observation's and action's type should be handled in the inherited classes)
        """
        raise NotImplementedError

    @abstractmethod
    def train_mode(self):
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self):
        raise NotImplementedError


class BCStochasticAgent(MultiTaskAgent):

    def __init__(self, **kwargs):
        self._actor = DiagGaussianActor(**kwargs)
        self._actor_optimizer = Adam(self._actor.parameters(), lr=0.0001)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._actor.to(self._device)

    def act_batch(self, obs: list, task: int) -> torch.Tensor:
        with torch.no_grad():
            dist = self._actor(obs, hlc=task)
            action = dist.mean
            action = action.clamp(-1, 1)
            return action

    def act_single(self, obs: dict, task: int) -> list:
        with torch.no_grad():
            dist = self._actor(obs, hlc=task)
            action = dist.mean
            action = action.clamp(-1, 1)
            action = list(to_np(action[0]))
            return action

    def update(self, obs: list, act: list, task: int) -> float:
        dist = self._actor(obs, hlc=task)
        act_e_hlc = torch.tensor(np.stack(act), device=self._device).float()
        log_prob_e = dist.log_prob(torch.clamp(act_e_hlc, min=-1 + 1e-6, max=1.0 - 1e-6)).sum(-1,
                                                                                              keepdim=True)
        bc_loss = - log_prob_e.mean()

        self._actor_optimizer.zero_grad()
        bc_loss.backward()
        self._actor_optimizer.step()

        return bc_loss.item()

    def train_mode(self):
        self._actor.train()

    def eval_mode(self):
        self._actor.eval()


class BCDeterministicAgent(MultiTaskAgent):
    def __init__(self, **kwargs):
        self._actor = Actor(kwargs["input_size"], kwargs["hidden_dim"], kwargs["action_dim"], output_factor=1)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._actor.to(self._device)
        self._actor_optimizer = Adam(self._actor.parameters(), lr=0.0001)
        self._actor_loss = nn.MSELoss()

    def act_batch(self, obs: list, task: int) -> Union[list, torch.Tensor]:
        with torch.no_grad():
            pred_act = self._actor(obs, task)
            return pred_act

    def act_single(self, obs: dict, task: int) -> list:
        with torch.no_grad():
            pred_act = self._actor(obs, task)
            return pred_act

    def update(self, obs: list, act: list, task: int) -> float:
        pred_act = self._actor(obs, task)
        act_e_hlc = torch.tensor(np.stack(act), device=self._device).float()
        mse_loss = self._actor_loss(pred_act, act_e_hlc)
        self._actor_optimizer.zero_grad()
        mse_loss.backward()
        self._actor_optimizer.step()
        return mse_loss.item()

    def train_mode(self):
        self._actor.train()

    def eval_mode(self):
        self._actor.eval()
