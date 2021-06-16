import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


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


class ConvReduction(nn.Module):
    """
    Takes a tensor with shape (B,C,H,W) and returns a tensor with shape (B,C). To achieve this two convolution layers
    are used.
    Output: (1028,)
    Input: (1028x4x4)
    """

    def __init__(self, in_channels: int):
        super(ConvReduction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(1024)
        # second convolution layer reduce the dimensional size to its half
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        # x.requires_grad = True
        return x


class Actor(nn.Module):
    """
    Output: a (3,)
    Input: s (1024x4x4)
    """
    HLCNUMBER_TO_HLC = {
        0: 'right',
        1: 'left',
        2: 'straight',
        3: 'follow_lane'
    }

    def __init__(self, hidden_size: int, action_dim: int = 2):
        super(Actor, self).__init__()
        self._device = 'cuda'
        self._input_channels = 1024
        self._action_dim = action_dim
        self._hidden_size = hidden_size
        self.conv_reduction = ConvReduction(self._input_channels)
        self.speed_mlp = TwoLayerMLP(2, 256, 128)

        input_size = 1152
        self.branches = torch.nn.ModuleDict({
            'left': TwoLayerMLP(input_size, hidden_size, self._action_dim * 2),
            'right': TwoLayerMLP(input_size, hidden_size, self._action_dim * 2),
            'follow_lane': TwoLayerMLP(input_size, hidden_size, self._action_dim * 2),
            'straight': TwoLayerMLP(input_size, hidden_size, self._action_dim * 2)
        })

    def forward(self, obs: Union[list, tuple, dict]):

        if isinstance(obs, list) or isinstance(obs, tuple):
            # if the observation is an iterable, then this method is going to be used for TRAINING in a batch-wise
            encoding = torch.stack([o['encoding'].to(self._device) for o in obs], dim=0)
            speed = torch.stack([o['speed'].to(self._device) for o in obs], dim=0).float()
            hlc = torch.stack([o['hlc'].to(self._device) for o in obs], dim=0).float()
        elif isinstance(obs, dict):
            # if the observation is a dict, then this method is going to be used for ACTION SELECTION
            encoding = obs['encoding'].unsqueeze(0).float()
            speed = torch.tensor(obs['speed'], device=self._device).unsqueeze(0).float()
            hlc = torch.tensor(obs['hlc'], device=self._device).unsqueeze(0).float()
        else:
            raise ValueError(f"Expected input of type list, tuple or dict but got: {type(obs)}")

        x = self.conv_reduction(encoding)  # B,1024,4,4 -> 1024
        speed_embedding = self.speed_mlp(speed)  # B,2, -> 128,
        x_speed = torch.cat([x, speed_embedding], dim=1)  # B,[1024, 128] -> B,1152

        # forward
        pred_branches = [self.branches[c](x_speed) for c in ["right", "left", "straight", "follow_lane"]]

        # unpack using hlc mask
        mask = hlc.long()
        pred_branches = torch.stack(pred_branches)
        pred_branches = torch.swapaxes(pred_branches, 0, -1)
        preds = torch.sum(F.one_hot(mask, num_classes=4) * pred_branches, dim=-1).T

        return preds


class Critic(nn.Module):
    """
    Output: Q(s,a): (1,)
    Input: (s, a); s: (1024x4x4); a: (3,)
    """

    def __init__(self, hidden_dim: int, action_dim: int, ):
        super(Critic, self).__init__()
        self._input_channels = 1024
        self._device = 'cuda'
        self.conv_reduction = ConvReduction(self._input_channels)
        self.speed_mlp = TwoLayerMLP(2, 256, 128)
        self.action_mlp = TwoLayerMLP(action_dim, 256, 128)

        input_size = 1280
        self.branches = torch.nn.ModuleDict({
            'left': TwoLayerMLP(input_size, hidden_dim, 1),
            'right': TwoLayerMLP(input_size, hidden_dim, 1),
            'follow_lane': TwoLayerMLP(input_size, hidden_dim, 1),
            'straight': TwoLayerMLP(input_size, hidden_dim, 1)
        })

    def forward(self, obs: Union[list, tuple], action: Union[list, tuple, torch.Tensor]):
        visual_embedding = torch.stack([o['visual'].to(self._device) for o in obs], dim=0)
        state = torch.stack([o['state'].to(self._device) for o in obs], dim=0).float()
        speed = state[:, :2]
        hlc = state[:, 2]

        if isinstance(action, list) or isinstance(action, tuple):
            action = torch.stack([torch.tensor(a) for a in action]).to(self._device)

        x = self.conv_reduction(visual_embedding)       # 1024x4x4 -> 1024
        speed_embedding = self.speed_mlp(speed)         # 2, -> 128,
        action_embedding = self.action_mlp(action)
        x_speed_action = torch.cat([x, speed_embedding, action_embedding], dim=1)  # [1024, 128, 128] -> 1280

        # TODO: Try to optimize this
        # forward
        pred_branches = []
        for c in ["right", "left", "straight", "follow_lane"]:
            pred_branches.append(self.branches[c](x_speed_action))

        # unpack using hlc mask
        mask = hlc.long()
        pred_branches = torch.stack(pred_branches)
        pred_branches = torch.swapaxes(pred_branches, 0, -1)
        preds = torch.sum(F.one_hot(mask, num_classes=4) * pred_branches, dim=-1).T
        preds = preds.squeeze(-1)
        return preds

#
# class SACCustom(SACTorchModel):
#     def __init__(self, *args, **kwargs):
#         super(SACCustom, self).__init__(*args, **kwargs)
#         self._actor = Actor(1024)
#         self._obs_space = args[0]
#
#     def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
#         model = Actor(1024)
#         return model
#
#     def build_q_model(self, obs_space, action_space, num_outputs,
#                       q_model_config, name):
#         model = Critic(1024)
#         return model
#
#     def get_policy_output(self, model_out: TensorType) -> TensorType:
#         state = model_out["state"]
#         speed = state[:, :2]
#         hlc = state[:, 3]
#         out = self.action_model(model_out["visual"], speed, hlc)
#         return out
#
#     def policy_variables(self):
#         return self.action_model.parameters()
#
#     def _get_q_value(self, model_out, actions, net):
#         pass
#
#     def q_variables(self):
#         return list(self.q_net.parameters()) + (list(self.twin_q_net.parameters()) if self.twin_q_net else [])


if __name__ == '__main__':
    batch_size = 8
    sample_input = torch.rand((batch_size, 1024, 4, 4))
    sample_speed = torch.rand((batch_size, 2))
    action = torch.tensor(np.random.random((batch_size, 3))).float()
    mse_loss = nn.MSELoss()

    critic = Critic(3, 512)
    actor = Actor(512, 3)
    q = critic(sample_input, sample_speed, action, "right")
    a = actor(sample_input, sample_speed, "right")

    expected_q = torch.rand((batch_size, 1))
    expected_a = torch.rand((batch_size, 3))
    a_loss = mse_loss(expected_a, a)
    q_loss = mse_loss(q, expected_q)

    q_loss.backward()
    a_loss.backward()
