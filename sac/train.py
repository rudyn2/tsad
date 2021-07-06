from termcolor import colored
import warnings
import wandb
import torch
from gym_carla.envs.carla_env import CarlaEnv
from models.ADEncoder import ADEncoder
from models.TemporalEncoder import VanillaRNNEncoder
from sac.agent.sac import SACAgent
from sac.agent.actor import DiagGaussianActor
from sac.agent.critic import DoubleQCritic
from sac.trainer import SACTrainer
from models.carla_wrapper import EncodeWrapper
from sac.replay_buffer import MixedReplayBuffer
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SAC Trainer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # encoder weight's
    parser.add_argument('--vis-weights', default='../dataset/weights/best_model_1_validation_accuracy=-0.5557.pt',
                        type=str, help="Path to visual encoder weight's")
    parser.add_argument('--temp-weights', default='../dataset/weights/best_VanillaRNNEncoder.pth')

    # in case of using behavioral cloning
    parser.add_argument('-i', '--input', required=False, default=None, type=str, help='path to hdf5 file')
    parser.add_argument('-m', '--metadata', required=False, default=None, type=str, help='path to json file')
    parser.add_argument('--debug', action='store_true', help='Whether or not visualize actor input')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # region: GENERAL PARAMETERS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    action_dim = 2
    actor_hidden_dim = 512
    critic_hidden_dim = 512

    online_memory_size = 512
    offline_dataset_path = None
    # endregion

    # region: init wandb
    wandb.init(project='tsad', entity='autonomous-driving')
    # endregion

    # region: init env
    print(colored("[*] Initializing models", "white"))
    visual = ADEncoder(backbone='efficientnet-b5')
    visual.load_state_dict(torch.load(args.vis_weights))
    visual.to(device)
    visual.freeze()

    temp = VanillaRNNEncoder()
    # temp.load_state_dict(torch.load(args.temp_weights))
    temp.to(device)
    temp.freeze()
    print(colored("[+] Encoder models were initialized and loaded successfully!", "green"))

    print(colored("[*] Initializing environment", "white"))
    env_params = {
        # carla connection parameters+
        'host': 'localhost',
        'port': 2000,  # connection port
        'town': 'Town01',  # which town to simulate

        # simulation parameters
        'verbose': False,
        'vehicles': 100,  # number of vehicles in the simulation
        'walkers': 0,  # number of walkers in the simulation
        'obs_size': 288,  # sensor width and height
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.025,  # time interval between two frames
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'max_time_episode': 400,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'speed_reduction_at_intersection': 0.75,
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    }
    carla_raw_env = CarlaEnv(env_params)
    carla_processed_env = EncodeWrapper(carla_raw_env, visual, temp, debug=args.debug)
    carla_processed_env.reset()
    print(colored("[+] Environment ready!", "green"))
    # endregion

    # region: init agent
    print(colored("[*] Initializing actor critic models", "white"))
    actor = DiagGaussianActor(action_dim=action_dim,
                              hidden_dim=actor_hidden_dim,
                              log_std_bounds=(-3, 3)
                              )
    critic = DoubleQCritic(action_dim=action_dim,
                           hidden_dim=critic_hidden_dim)
    target_critic = DoubleQCritic(action_dim=action_dim,
                                  hidden_dim=critic_hidden_dim)
    agent = SACAgent(actor=actor,
                     critic=critic,
                     target_critic=target_critic,
                     action_dim=action_dim)
    print(colored("[*] SAC Agent is ready!", "green"))
    # endregion

    # region: init buffer
    print(colored("[*] Initializing Mixed Replay Buffer", "white"))
    if offline_dataset_path:
        print(colored("Full DRL mode"))
        mixed_replay_buffer = MixedReplayBuffer(online_memory_size,
                                                offline_buffer_hdf5=str(offline_dataset_path) + '.hdf5',
                                                offline_buffer_json=str(offline_dataset_path) + '.json')
    else:
        print(colored("BC + DRL mode"))
        mixed_replay_buffer = MixedReplayBuffer(online_memory_size)
    print(colored("[*] Initializing Mixed Replay Buffer", "green"))
    # endregion

    train_params = {
        "device": "cuda",
        "seed": 42,
        "log_save_tb": True,
        "num_eval_episodes": 5,     # number of episodes used for evaluation
        "num_train_steps": 1e6,     # number of training steps
        "eval_frequency": 10000,    # number of steps required for evaluation
        "num_seed_steps": 200   # number of steps before starting to update the models
    }
    print(colored("Training", "white"))
    trainer = SACTrainer(env=carla_processed_env,
                         agent=agent,
                         buffer=mixed_replay_buffer,
                         **train_params
                         )
    try:
        trainer.run()
    finally:
        trainer.end()
