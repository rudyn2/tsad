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


class Actor(nn.Module):
    """
    Output: a (3,)
    Input: s (1024x4x4)
    """

    def __init__(self, hidden_size: int, action_dim: int = 2):
        super(Actor, self).__init__()
        self._device = 'cuda'
        self._action_dim = action_dim
        self._hidden_size = hidden_size
        self.input_mlp = TwoLayerMLP(15, 128, 64)
        self.speed_mlp = TwoLayerMLP(3, 128, 64)

        input_size = 64 + 64
        self.branches = torch.nn.ModuleDict({
            'left': TwoLayerMLP(input_size, hidden_size, self._action_dim * 2),
            'right': TwoLayerMLP(input_size, hidden_size, self._action_dim * 2),
            'follow_lane': TwoLayerMLP(input_size, hidden_size, self._action_dim * 2),
            'straight': TwoLayerMLP(input_size, hidden_size, self._action_dim * 2)
        })

    def forward(self, obs: Union[list, tuple, dict], hlc):

        if isinstance(obs, list) or isinstance(obs, tuple):
            # if the observation is an iterable, then this method is going to be used for TRAINING in a batch-wise
            encoding = torch.stack([torch.tensor(o['encoding'], device=self._device) for o in obs], dim=0).float()
            speed = torch.stack([torch.tensor(o['speed'], device=self._device) for o in obs], dim=0).float()
        elif isinstance(obs, dict):
            # if the observation is a dict, then this method is going to be used for ACTION SELECTION
            encoding = torch.tensor(obs['encoding'], device=self._device).unsqueeze(0).float()
            speed = torch.tensor(obs['speed'], device=self._device).unsqueeze(0).float()
        else:
            raise ValueError(f"Expected input of type list, tuple or dict but got: {type(obs)}")

        encoding = self.input_mlp(encoding)
        speed_embedding = self.speed_mlp(speed)  # B,2, -> 128,
        x_speed = torch.cat([encoding, speed_embedding], dim=1)  # B,[1024, 128] -> B,1152

        # forward
        preds = self.branches[__HLCNUMBER_TO_HLC__[hlc]](x_speed)
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
        self.input_mlp = TwoLayerMLP(15, 128, 64)
        self.speed_mlp = TwoLayerMLP(3, 128, 64)
        self.action_mlp = TwoLayerMLP(action_dim, 128, 64)

        input_size = 64 + 64 + 64
        self.branches = torch.nn.ModuleDict({
            'left': TwoLayerMLP(input_size, hidden_dim, 1),
            'right': TwoLayerMLP(input_size, hidden_dim, 1),
            'follow_lane': TwoLayerMLP(input_size, hidden_dim, 1),
            'straight': TwoLayerMLP(input_size, hidden_dim, 1)
        })

    def forward(self, obs: Union[list, tuple], action: Union[list, tuple, torch.Tensor], hlc: int):
        encoding = torch.stack([torch.tensor(o['encoding'], device=self._device) for o in obs], dim=0).float()
        speed = torch.stack([torch.tensor(o['speed'], device=self._device) for o in obs], dim=0).float()

        if isinstance(action, list) or isinstance(action, tuple):
            action = torch.stack([torch.tensor(a) for a in action]).to(self._device)

        encoding = self.input_mlp(encoding)
        speed_embedding = self.speed_mlp(speed)
        action_embedding = self.action_mlp(action)
        x_speed_action = torch.cat([encoding, speed_embedding, action_embedding], dim=1)  # [1024, 128, 128] -> 1280

        # forward
        preds = self.branches[__HLCNUMBER_TO_HLC__[hlc]](x_speed_action)
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
