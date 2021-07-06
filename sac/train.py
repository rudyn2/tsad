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
    # carla parameters
    carla_config = parser.add_argument_group('CARLA config')
    carla_config.add_argument('--host', default='172.18.0.1', type=str, help='IP address of CARLA host.')
    carla_config.add_argument('--port', default=2008, type=int, help='Port number of CARLA host.')
    carla_config.add_argument('--vehicles', default=100, type=int, help='Number of vehicles in the simulation.')
    carla_config.add_argument('--walkers', default=50, type=int, help='Number of walkers in the simulation.')

    # encoder weight's
    encoders_group = parser.add_argument_group('Encoders config')
    parser.add_argument('--vis-weights', default='../dataset/weights/best_model_1_validation_accuracy=-0.5557.pt',
                        type=str, help="Path to visual encoder weight's")
    parser.add_argument('--temp-weights', default='../dataset/weights/best_VanillaRNNEncoder(2).pth',
                        help='Path to temporal encoder weights')

    # SAC parameters
    rl_group = parser.add_argument_group('RL Config')
    rl_group.add_argument('--num-seed', default=2000, type=int, help='Number of seed steps before starting to train.')
    rl_group.add_argument('--max-episode-steps', default=200, type=int, help='Maximum number of steps per episode.')
    rl_group.add_argument('--num-eval-episodes', default=5, type=int, help='Number of evaluation episodes.')
    rl_group.add_argument('--num-train-steps', default=1e6, type=int, help='Number of training steps.')
    rl_group.add_argument('--eval-frequency', default=10000, type=int, help='number of steps between evaluations')

    models_parameters = parser.add_argument_group('Actor-Critic config')
    models_parameters.add_argument('--actor-hidden-dim', type=int, default=512, help='Size of hidden layer in the '
                                                                                     'actor model.')
    models_parameters.add_argument('--critic-hidden-dim', type=int, default=512, help='Size of hidden layer in the '
                                                                                      'critic model.')

    buffer_group = parser.add_argument_group('Buffer config')
    buffer_group.add_argument('--batch-size', default=1024, type=int, help='Batch size.')
    buffer_group.add_argument('--bc-proportion', default=0.25, type=float,
                              help='Batch size proportion of offline samples that are going to be used for '
                                   'behavioral cloning.')
    buffer_group.add_argument('--online-memory-size', default=512, type=int, help='Number of steps to be stored in the'
                                                                                  'online buffer')

    # in case of using behavioral cloning
    bc_group = parser.add_argument_group('Behavioral cloning config')
    bc_group.add_argument('--bc', default=None, type=str, help='path to dataset (without extensions)')
    bc_group.add_argument('--debug', action='store_true', help='Whether or not visualize actor input')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # region: GENERAL PARAMETERS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    action_dim = 2
    offline_dataset_path = args.bc
    # endregion

    # region: init wandb
    wandb.init(project='tsad', entity='autonomous-driving')
    # endregion

    # region: init env
    print(colored("[*] Initializing models", "white"))
    visual = ADEncoder(backbone='efficientnet-b5')
    visual.load_state_dict(torch.load(args.vis_weights))
    visual.to(device)
    visual.eval()
    visual.freeze()

    temp = VanillaRNNEncoder(num_layers=4,
                             hidden_size=1024,
                             action__chn=256,
                             speed_chn=256,
                             bidirectional=True)
    temp.load_state_dict(torch.load(args.temp_weights))
    temp.to(device)
    temp.eval()
    temp.freeze()
    print(colored("[+] Encoder models were initialized and loaded successfully!", "green"))

    print(colored("[*] Initializing environment", "white"))
    env_params = {
        # carla connection parameters+
        'host': args.host,
        'port': args.port,  # connection port
        'town': 'Town01',  # which town to simulate

        # simulation parameters
        'verbose': False,
        'vehicles': args.vehicles,  # number of vehicles in the simulation
        'walkers': args.walkers,  # number of walkers in the simulation
        'obs_size': 288,  # sensor width and height
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.025,  # time interval between two frames
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'max_time_episode': args.max_episode_steps,  # maximum timesteps per episode
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
                              hidden_dim=args.actor_hidden_dim,
                              log_std_bounds=(-3, 3)
                              )
    critic = DoubleQCritic(action_dim=action_dim,
                           hidden_dim=args.critic_hidden_dim)
    target_critic = DoubleQCritic(action_dim=action_dim,
                                  hidden_dim=args.critic_hidden_dim)
    agent = SACAgent(actor=actor,
                     critic=critic,
                     target_critic=target_critic,
                     action_dim=action_dim,
                     batch_size=args.batch_size,
                     offline_proportion=args.bc_proportion)
    print(colored("[*] SAC Agent is ready!", "green"))
    # endregion

    # region: init buffer
    print(colored("[*] Initializing Mixed Replay Buffer", "white"))
    if offline_dataset_path:
        print(colored("Full DRL mode"))
        mixed_replay_buffer = MixedReplayBuffer(args.online_memory_size,
                                                offline_buffer_hdf5=str(offline_dataset_path) + '.hdf5',
                                                offline_buffer_json=str(offline_dataset_path) + '.json')
    else:
        print(colored("BC + DRL mode"))
        mixed_replay_buffer = MixedReplayBuffer(args.online_memory_size)
    print(colored("[*] Initializing Mixed Replay Buffer", "green"))
    # endregion

    train_params = {
        "device": "cuda",
        "seed": 42,
        "log_save_tb": True,
        "num_eval_episodes": 5,  # number of episodes used for evaluation
        "num_train_steps": 1e6,  # number of training steps
        "eval_frequency": 10000,  # number of steps required for evaluation
        "num_seed_steps": args.num_seed  # number of steps before starting to update the models
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
