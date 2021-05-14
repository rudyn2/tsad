import ray
import ray.rllib.agents.sac as sac

from models.carla_wrapper import EncodeWrapper


ray.init()
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
trainer = sac.SACTrainer(env=EncodeWrapper,
                         config={
                             "env_config": params,
                             "model": {
                                 "custom_model": "my_tf_model",
                                 # Extra kwargs to be passed to your model's c'tor.
                                 "custom_model_config": {},
                             },
                         })
