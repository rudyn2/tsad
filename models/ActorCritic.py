import torch.nn
import torch.nn as nn
import numpy as np


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


class SEBlock(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


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
        return x


class Actor(nn.Module):
    """
    Output: a (3,)
    Input: s (1024x4x4)
    """

    def __init__(self, input_channels: int):
        super(Actor, self).__init__()
        self._input_channels = input_channels
        self.conv_reduction = ConvReduction(self._input_channels)
        self.speed_mlp = TwoLayerMLP(2, 256, 128)

        input_size = 1152
        self.branches = torch.nn.ModuleDict({
            'left': TwoLayerMLP(input_size, 512, 3),
            'right': TwoLayerMLP(input_size, 512, 3),
            'follow_lane': TwoLayerMLP(input_size, 512, 3),
            'straight': TwoLayerMLP(input_size, 512, 3)
        })

    def forward(self, state: torch.Tensor, speed: torch.Tensor, hlc: str):
        x = self.conv_reduction(state)                                      # 1024x4x4 -> 1024
        speed_embedding = self.speed_mlp(speed)       # 2, -> 128,
        x_speed = torch.cat([x, speed_embedding], dim=1)                    # [1024, 128] -> 1152
        x = self.branches[hlc](x_speed)                                     # 3,
        return x


class Critic(nn.Module):
    """
    Output: Q(s,a): (1,)
    Input: (s, a); s: (1024x4x4); a: (3,)
    """

    def __init__(self, input_channels: int):
        super(Critic, self).__init__()
        self._input_channels = input_channels
        self.conv_reduction = ConvReduction(self._input_channels)
        self.speed_mlp = TwoLayerMLP(2, 256, 128)
        self.action_mlp = TwoLayerMLP(3, 256, 128)

        input_size = 1280
        self.branches = torch.nn.ModuleDict({
            'left': TwoLayerMLP(input_size, 512, 1),
            'right': TwoLayerMLP(input_size, 512, 1),
            'follow_lane': TwoLayerMLP(input_size, 512, 1),
            'straight': TwoLayerMLP(input_size, 512, 1)
        })

    def forward(self, state: torch.Tensor, speed: torch.Tensor, action: torch.Tensor, hlc: str):
        x = self.conv_reduction(state)                                      # 1024x4x4 -> 1024
        speed_embedding = self.speed_mlp(speed)       # 2, -> 128,
        action_embedding = self.action_mlp(action)
        x_speed = torch.cat([x, speed_embedding, action_embedding], dim=1)  # [1024, 128, 128] -> 1280
        x = self.branches[hlc](x_speed)                                     # 3,
        return x


if __name__ == '__main__':
    batch_size = 8
    sample_input = torch.rand((batch_size, 1024, 4, 4))
    sample_speed = torch.rand((batch_size, 2))
    action = torch.tensor(np.random.random((batch_size, 3))).float()
    mse_loss = nn.MSELoss()

    critic = Critic(input_channels=1024)
    actor = Actor(input_channels=1024)
    q = critic(sample_input, sample_speed, action, "right")
    a = actor(sample_input, sample_speed, "right")

    expected_q = torch.rand((batch_size, 1))
    expected_a = torch.rand((batch_size, 3))
    a_loss = mse_loss(expected_a, a)
    q_loss = mse_loss(q, expected_q)

    q_loss.backward()
    a_loss.backward()

