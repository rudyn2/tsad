import os
import random

import numpy as np
import torch
from torch import nn

from models.carla_wrapper import DummyWrapper


def make_env(cfg):
    """Helper function to create dm_control environment"""
    env = DummyWrapper(cfg)

    return env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def calc_reward(metadata: dict, speed_red: float = 0.75, desired_speed: float = 6) -> float:
    steer = metadata['control']['steer']
    command = metadata['command']
    distance = metadata['lane_distance']
    collision = None

    # speed and steer behavior
    if command in ['RIGHT', 'LEFT']:
        r_a = 2 - np.abs(speed_red * desired_speed - metadata['speed']) / speed_red * desired_speed
        is_opposite = steer > 0 and command == 'LEFT' or steer < 0 and command == 'RIGHT'
        r_a -= steer ** 2 if is_opposite else 0
    elif command == 'STRAIGHT':
        r_a = 1 - np.abs(speed_red * desired_speed - metadata['speed']) / speed_red * desired_speed
    # follow lane
    else:
        r_a = 2 - np.abs(desired_speed - metadata['speed']) / desired_speed

    # collision
    r_c = 0
    if collision:
        r_c = -5
        if str(collision['other_actor']).startswith('vehicle'):
            r_c = -10

    # distance to center
    r_dist = - distance / 2
    return r_a + r_c + r_dist