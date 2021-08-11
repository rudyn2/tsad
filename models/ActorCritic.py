import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

__HLCNUMBER_TO_HLC__ = {
    0: 'right',
    1: 'left',
    2: 'straight',
    3: 'follow_lane'
}


class TwoLayerMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_size: int, output_size: int):
        super(TwoLayerMLP, self).__init__()
        self._input_dim = input_dim
        self._hidden_size = hidden_size
        self._output_size = output_size
        self.fc1 = nn.Linear(self._input_dim, self._hidden_size)
        self.fc2 = nn.Linear(self._hidden_size, self._output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ThreeLayerMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_sizes: tuple, output_size: int):
        super(ThreeLayerMLP, self).__init__()
        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._output_size = output_size
        self.fc1 = nn.Linear(self._input_dim, self._hidden_sizes[0])
        self.fc2 = nn.Linear(self._hidden_sizes[0], self._hidden_sizes[1])
        self.fc3 = nn.Linear(self._hidden_sizes[1], self._output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """
    Output: a (3,)
    Input: s (1024x4x4)
    """

    def __init__(self, hidden_size: int, action_dim: int = 2, output_factor: int = 2):
        super(Actor, self).__init__()
        self._device = 'cuda'
        self._action_dim = action_dim
        self._hidden_size = hidden_size

        input_size = 15
        self.branches = torch.nn.ModuleDict({
            'left': ThreeLayerMLP(input_size, (hidden_size, hidden_size // 2), self._action_dim * output_factor),
            'right': ThreeLayerMLP(input_size, (hidden_size, hidden_size // 2), self._action_dim * output_factor),
            'follow_lane': ThreeLayerMLP(input_size, (hidden_size, hidden_size // 2), self._action_dim * output_factor),
            'straight': ThreeLayerMLP(input_size, (hidden_size, hidden_size // 2), self._action_dim * output_factor)
        })

    def forward(self, obs: Union[list, tuple, dict], hlc):

        if isinstance(obs, list) or isinstance(obs, tuple):
            # if the observation is an iterable, then this method is going to be used for TRAINING in a batch-wise
            encoding = torch.stack([torch.tensor(o['encoding'], device=self._device) for o in obs], dim=0).float()
        elif isinstance(obs, dict):
            # if the observation is a dict, then this method is going to be used for ACTION SELECTION
            encoding = torch.tensor(obs['encoding'], device=self._device).unsqueeze(0).float()
        else:
            raise ValueError(f"Expected input of type list, tuple or dict but got: {type(obs)}")

        # forward
        preds = self.branches[__HLCNUMBER_TO_HLC__[hlc]](encoding)
        return preds


class Critic(nn.Module):
    """
    Output: Q(s,a): (1,)
    Input: (s, a); s: (1024x4x4); a: (3,)
    """

    def __init__(self, hidden_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self._input_channels = 768
        self._device = 'cuda'

        input_size = 15 + action_dim
        self.branches = torch.nn.ModuleDict({
            'left': ThreeLayerMLP(input_size, (hidden_dim, hidden_dim // 2), 1),
            'right': ThreeLayerMLP(input_size, (hidden_dim, hidden_dim // 2), 1),
            'follow_lane': ThreeLayerMLP(input_size, (hidden_dim, hidden_dim // 2), 1),
            'straight': ThreeLayerMLP(input_size, (hidden_dim, hidden_dim // 2), 1)
        })

    def forward(self, obs: Union[list, tuple], action: Union[list, tuple, torch.Tensor], hlc: int):
        encoding = torch.stack([torch.tensor(o['encoding'], device=self._device) for o in obs], dim=0).float()

        if isinstance(action, list) or isinstance(action, tuple):
            action = torch.stack([torch.tensor(a) for a in action]).to(self._device)
        
        x_action = torch.cat([encoding, action], dim=1)

        # forward
        preds = self.branches[__HLCNUMBER_TO_HLC__[hlc]](x_action)
        return preds


if __name__ == '__main__':
    # DEPRECATED -->
    batch_size = 8
    sample_input = torch.rand((batch_size, 15))
    sample_speed = torch.rand((batch_size, 2))
    action = torch.tensor(np.random.random((batch_size, 3))).float()
    mse_loss = nn.MSELoss()

    critic = Critic(512, 3)
    actor = Actor(512, 3)
    q = critic(sample_input, sample_speed, action, "right")
    a = actor(sample_input, sample_speed, "right")

    expected_q = torch.rand((batch_size, 1))
    expected_a = torch.rand((batch_size, 3))
    a_loss = mse_loss(expected_a, a)
    q_loss = mse_loss(q, expected_q)

    q_loss.backward()
    a_loss.backward()
    # <---
