import gym
import torch.nn
import numpy as np


class EncodeWrapper(gym.core.ObservationWrapper):

    def __init__(self, env, visual_encoder: torch.nn.Module, temporal_encoder: torch.nn.Module):
        super(EncodeWrapper, self).__init__(env)
        self._visual_encoder = visual_encoder
        self._temporal_encoder = temporal_encoder

    def observation(self, observation):
        rgb = torch.tensor(observation['camera'])
        depth = torch.tensor(observation['depth'])
        x = np.concatenate((rgb, depth[:, :, np.newaxis]), axis=2)
        x = np.transpose(x, axes=[2, 0, 1])
        x = torch.tensor(x)
        visual_embedding = self._visual_encoder.encode(x.unsqueeze(0).float())
        # temporal_embedding = self._temporal_encoder(visual_embedding)
        return visual_embedding


if __name__ == '__main__':
    from gym_carla.envs.carla_env import CarlaEnv
    from ADEncoder import ADEncoder
    from TemporalEncoder import RNNEncoder
    visual = ADEncoder(backbone='efficientnet-b5')
    temp = RNNEncoder()

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
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    }
    carla_raw_env = CarlaEnv(params)
    carla_processed_env = EncodeWrapper(carla_raw_env, visual, temp)
    carla_processed_env.reset()
