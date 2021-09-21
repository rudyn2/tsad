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
    def act_batch(self, obs: torch.Tensor, task: int) -> Union[list, torch.Tensor]:
        """
        Given a list of observations in the context of some task, return the predicted action for each observation.
        (the observation's type should be handled in the inherited classes)
        """
        raise NotImplementedError

    @abstractmethod
    def act_single(self, obs: torch.Tensor, sample: bool, task: int) -> list:
        """
        Given a single observation in the context of some task, return the predicted action.
        If sample is False, the actor must return the mean of the predicted distribution. Otherwise,
        a sample from that distribution is returned.
        (the observation's type should be handled in the inherited classes)
        """
        raise NotImplementedError

    @abstractmethod
    def get_supervised_loss(self, obs: torch.Tensor, act: torch.Tensor, task: int) -> torch.Tensor:
        """
        Given a list of observations in the context of some task, return the supervised loss.
        (the observation's and action's type should be handled in the inherited classes)
        """
        raise NotImplementedError


class MultiTaskCritic(MultiTaskAgent, ABC):
    def __init__(self):
        super(MultiTaskCritic, self).__init__()

