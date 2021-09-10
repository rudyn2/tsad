from gym_carla.envs.carla_env import CarlaEnv
from typing import Dict, Tuple
from gym.core import Wrapper


class EncodeWrapper(Wrapper):

    def __init__(self,
                 env: CarlaEnv,
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

        self.step_ = 0
        self._visual_buffer = None
        self._last_speed = None
        self._debug = debug

    def reset(self, **kwargs):
        """
        Warm-up. Fill the buffer with the last 4 observations.
        """
        obs = self.env.reset()
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
        self._last_speed = observation['speed']

        obs = {
            "encoding": observation['affordances'],
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
    carla_processed_env = EncodeWrapper(carla_raw_env)
    obs = carla_processed_env.reset()
    for i in range(100):
        obs, reward, done, info = carla_processed_env.step([1, 0, 0])
        print(f"step={i}, encoding shape: {obs['encoding'].shape}, rew={reward:.2f}")
        if done:
            carla_processed_env.reset()
            break
