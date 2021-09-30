from abc import abstractmethod
from typing import Union

import numpy as np
import torch.cuda
import torch.nn as nn
from torch.optim.adam import Adam

from sac.agent.actor import DiagGaussianActor
from models.ActorCritic import Actor
from squashed_gaussian import SquashedGaussianMLPActor


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
    
    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError


class BCStochasticAgent(MultiTaskAgent):

    def __init__(self, **kwargs):
        self._actor = nn.ModuleDict({
            str(hlc): SquashedGaussianMLPActor(
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
            action = self._actor[str(task)](encoding, deterministic=True)
            return action

    def act_single(self, obs: dict, task: int) -> list:
        # obs = {'encoding': ..., 'hlc': ...}
        # task = 3
        encodings = torch.tensor(obs['affordances'], device=self._device).unsqueeze(0).float()
        with torch.no_grad():
            action = self._actor[str(task)](encodings, deterministic=True)  # (1, 2)
        return list(action.squeeze(dim=0).cpu().numpy())                    # [target_speed, steer]

    def update(self, obs: list, act: list, task: int, speed = None) -> float:
        encoding = torch.stack([torch.tensor(o['encoding'], device=self._device) for o in obs], dim=0).float()
        act = torch.tensor(act, device=self._device).float()
        if speed:
            pred_act, _, pred_speed = self._actor[str(task)].get_distribution_with_speed(encoding)
            loss = self._mse(pred_act, act) + self._mse(pred_speed, speed)
        else:
            pred_act, _ = self._actor[str(task)].get_distribution(encoding)
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


class BCDeterministicAgent(MultiTaskAgent):
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
