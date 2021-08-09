import numpy as np
import torch.nn
from gym_carla.envs.carla_env import CarlaEnv
from typing import Dict, Tuple
from torch import Tensor
from scripts.visualize import labels_to_cityscapes_palette
from gym.core import Wrapper
import torchvision.transforms as T
import cv2


class EncodeWrapper(Wrapper):

    def __init__(self,
                 env: CarlaEnv,
                 visual_encoder: torch.nn.Module,
                 temporal_encoder: torch.nn.Module,
                 max_steps: int = 200,
                 action_frequency: int = 4,
                 reward_scale: float = 1,
                 device: str = 'cuda',
                 debug: bool = True):
        self._action_frequency = action_frequency
        env.max_steps = max_steps * action_frequency
        super(EncodeWrapper, self).__init__(env)

        self._reward_scale = reward_scale
        self._device = device
        self._to_tensor = T.ToTensor()

        self.step_ = 0
        self._visual_buffer = None
        self._hidden_temp = self._temporal_encoder.init_hidden(1, device=self._device)
        self._last_speed = None
        self._debug = debug

    def _format_observation(self, observation: Dict[str, np.ndarray]):
        """
        Takes an observation of an rgb image and a depth map and returns a stacked torch tensor
        """
        rgb = self._to_tensor(observation['camera'])
        depth = torch.tensor(observation['depth']) / 1000
        x = torch.cat([depth.unsqueeze(0), rgb])
        x = x.unsqueeze(0).to(self._device).float()
        return x

    def reset(self, **kwargs):
        """
        Warm-up. Fill the buffer with the last 4 observations.
        """
        obs = self.env.reset()

        encoded_obs = self.create_visual_encoding(obs)
        self._visual_buffer = encoded_obs
        self._hidden_temp = self._temporal_encoder.init_hidden(1, device=self._device)
        self._last_speed = obs['speed']

        return self._step([0, 0, 0])[0]

    def _step(self, action: list) -> Tuple[Dict, float, bool, dict]:
        """
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # do an step
        observation, reward, done, info = self.env.step(action)
        reward = self._reward_scale * reward

        # create actual encoding using provided action
        hidden_state, temporal_encoding = self.create_temporal_encoding(action)  # (1, 512, 4, 4)

        # save new state (next observation)
        visual_encoding = self.create_visual_encoding(observation)  # (1, 512, 4, 4)
        self._visual_buffer = visual_encoding
        self._hidden_temp = hidden_state
        self._last_speed = observation['speed']

        encoding = torch.cat([visual_encoding, temporal_encoding], dim=1).squeeze(0)

        obs = {
            "encoding": encoding,
            "speed": self._last_speed,
            "hlc": observation['hlc'] - 1
        }

        # OVERRIDE DONE SIGNAL IN CASE OF HLC != LANE_FOLLOW
        if obs["hlc"] != 3:
            done = True

        return obs, reward, done, info

    def step(self, action: list):
        # repeat the action 4 times
        for _ in range(self._action_frequency - 1):
            self._step(action)
        return self._step(action)


if __name__ == '__main__':
    from ADEncoder import ADEncoder
    from TemporalEncoder import SequenceRNNEncoder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vis_weights = r'C:\Users\C0101\Documents\tsad\dataset\weights\mobilenet_final_weights.pt'
    temp_weights = r'C:\Users\C0101\Documents\tsad\dataset\weights\best_SequenceRNNEncoder.pth'

    visual = ADEncoder(backbone='mobilenetv3_small_075')
    visual.load_state_dict(torch.load(vis_weights))
    visual.to(device)
    visual.eval()
    visual.freeze()

    temp = SequenceRNNEncoder(num_layers=2,
                              hidden_size=1024,
                              action__chn=1024,
                              speed_chn=1024,
                              bidirectional=True)
    temp.load_state_dict(torch.load(temp_weights))
    temp.to(device)
    temp.eval()
    temp.freeze()

    params = {
        # carla connection parameters+
        'host': 'localhost',
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate

        # simulation parameters
        'verbose': False,
        'vehicles': 100,  # number of vehicles in the simulation
        'walkers': 0,  # number of walkers in the simulation
        'obs_size': 288,  # sensor width and height
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.025,  # time interval between two frames
        'reward_weights': (0.3, 0.3, 0.3),
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'speed_reduction_at_intersection': 0.75,
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    }
    carla_raw_env = CarlaEnv(params)
    carla_processed_env = EncodeWrapper(carla_raw_env, visual, temp)
    obs = carla_processed_env.reset()
    for i in range(100):
        obs, reward, done, info = carla_processed_env.step([1, 0, 0])
        print(f"step={i}, encoding shape: {obs['encoding'].shape}, rew={reward:.2f}")
        if done:
            carla_processed_env.reset()
            break
