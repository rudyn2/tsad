from typing import Union
from abc import abstractmethod, ABC
import torch


class MultiTaskAgent(ABC):

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


class MultiTaskActor(MultiTaskAgent, ABC):

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
    def supervised_update(self, obs: list, act: list, task: int) -> float:
        """
        Given a list of observations in the context of some task, update the agent and return some metric (e.g. loss).
        (the observation's and action's type should be handled in the inherited classes)
        """
        raise NotImplementedError


class MultiTaskCritic(MultiTaskAgent, ABC):
    def __init__(self):
        super(MultiTaskCritic, self).__init__()

