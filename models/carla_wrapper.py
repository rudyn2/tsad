import gym
from gym import spaces
import numpy as np
import torch.nn
import random
from gym_carla.envs import CarlaEnv
from typing import Dict
from torch import Tensor
from collections import deque
from gym.core import Wrapper


class EncodeWrapper(Wrapper):

    def __init__(self,
                 env: CarlaEnv,
                 visual_encoder: torch.nn.Module,
                 temporal_encoder: torch.nn.Module,
                 device: str = 'cuda'):
        super(EncodeWrapper, self).__init__(env)

        self._device = device
        self._visual_encoder = visual_encoder
        self._temporal_encoder = temporal_encoder

        self._visual_encoder.to(self._device)
        self._temporal_encoder.to(self._device)

        self._visual_buffer = deque(maxlen=4)
        self._last_speed = None

    def _format_observation(self, observation: Dict[str, np.ndarray]):
        """
        Takes an observation of an rgb image and a depth map and returns a stacked torch tensor
        """
        rgb = torch.tensor(observation['camera'])
        depth = torch.tensor(observation['depth'])
        x = np.concatenate((rgb, depth[:, :, np.newaxis]), axis=2)
        x = np.transpose(x, axes=[2, 0, 1])
        x = torch.tensor(x, device=self._device).unsqueeze(0).float()
        return x

    def reset(self, **kwargs):
        """
        Warm-up. Fill the buffer with the last 4 observations.
        """
        self._visual_buffer.clear()
        obs = self.env.reset()

        # load 4 frames (needed for temporal encoder)
        encoded_obs = self.create_visual_encoding(obs)
        self._visual_buffer.append(encoded_obs)
        for _ in range(3):
            # execute a neutral action to load buffer
            obs, _, _, _ = self.env.step([0, 0])

            # encode received observation
            encoded_obs = self.create_visual_encoding(obs)

            # save in buffer and update last speed
            self._last_speed = obs['state'][:2]
            self._visual_buffer.append(encoded_obs)

        # neutral step to begin
        # this is needed to include the temporal encoding
        return self.step([0, 0])[0]

    def step(self, action: list):
        """
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # do an step
        observation, reward, done, info = self.env.step(action)

        # create actual encoding using provided action
        temporal_encoding = self.create_temporal_encoding(action)  # (1, 512, 4, 4)

        # save new state (next observation)
        visual_encoding = self.create_visual_encoding(observation)  # (1, 512, 4, 4)
        self._visual_buffer.append(visual_encoding)
        self._last_speed = np.array(observation['state'][:2])

        encoding = torch.cat([visual_encoding.to(self._device),
                              temporal_encoding], dim=1).squeeze(0)

        return encoding, reward, done, info

    def create_temporal_encoding(self, action: list) -> Tensor:
        """
        Given the last 4 simulation steps, it creates an temporal encoding using provided RNN. In order to achieve that,
        the RNN is trying to predict the next simulation step.
        """
        stacked_frames = torch.cat([self._visual_buffer[0],
                                    self._visual_buffer[1],
                                    self._visual_buffer[2],
                                    self._visual_buffer[3]], dim=0).unsqueeze(0).float()
        stacked_frames = stacked_frames.to(self._device)
        action = torch.tensor(action, device=self._device).unsqueeze(0).float()
        speed = torch.tensor(self._last_speed, device=self._device).unsqueeze(0).float()
        lengths = torch.tensor([4], device='cpu').long()

        temporal_encoding = self._temporal_encoder.forward(stacked_frames, action, speed, lengths)
        return temporal_encoding

    def create_visual_encoding(self, observation: Dict[str, np.ndarray]) -> Tensor:
        obs = self._format_observation(observation)
        encoded_obs = self._visual_encoder.encode(obs)
        encoded_obs = encoded_obs.detach().cpu()
        return encoded_obs


class DummyWrapper(gym.Env):

    def __init__(self, config):
        super(DummyWrapper, self).__init__()
        self.steps = 0
        self.end_step = config['steps']
        self._max_episode_steps = config['steps']
        self.action_space = spaces.Box(np.array([0, 1]),
                                       np.array([-1, 1]),
                                       dtype=np.float32)  # acc, steer
        observation_space_dict = {
            'visual': spaces.Box(low=0, high=1, shape=(1024, 4, 4), dtype=np.float32),
            'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
        }

        self.observation_space = spaces.Dict(observation_space_dict)

    def render(self, mode='human'):
        pass

    def step(self, action):
        self.steps += 1
        done = self.steps == self.end_step
        return {"visual": torch.rand((1024, 4, 4)),
                "state": torch.randint(0, high=4, size=(3,))}, random.random(), done, {}

    def reset(self, **kwargs):
        self.steps = 0
        return {"visual": torch.rand((1024, 4, 4)),
                "state": torch.randint(0, high=4, size=(3,))}

    @property
    def max_episode_steps(self):
        return self._max_episode_steps


if __name__ == '__main__':
    from gym_carla.envs.carla_env import CarlaEnv
    from ADEncoder import ADEncoder
    from TemporalEncoder import VanillaRNNEncoder

    visual = ADEncoder(backbone='efficientnet-b5')
    temp = VanillaRNNEncoder()

    params = {
        'number_of_vehicles': 100,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.05,  # time interval between two frames
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'reduction_at_intersection': 0.75
    }
    carla_raw_env = CarlaEnv(params)
    carla_processed_env = EncodeWrapper(carla_raw_env, visual, temp)
    obs = carla_processed_env.reset()
    for _ in range(2):
        obs, reward, done, info = carla_processed_env.step([0, 0])
        print(obs.shape, reward, done)
