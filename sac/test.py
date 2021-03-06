from termcolor import colored
import warnings
import wandb
import torch
from gym_carla.envs.carla_env import CarlaEnv
from sac.sac import SACAgent
from sac.actor import DiagGaussianActor
from sac.critic import DoubleQCritic
from sac.trainer import SACTrainer
from models.carla_wrapper import EncodeWrapper
import argparse
import traceback

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAC Trainer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # carla parameters
    carla_config = parser.add_argument_group('CARLA config')
    carla_config.add_argument('--host', default='172.18.0.1', type=str, help='IP address of CARLA host.')
    carla_config.add_argument('--port', default=2008, type=int, help='Port number of CARLA host.')
    carla_config.add_argument('--vehicles', default=100, type=int, help='Number of vehicles in the simulation.')
    carla_config.add_argument('--walkers', default=50, type=int, help='Number of walkers in the simulation.')

    # SAC wights
    carla_config = parser.add_argument_group('SAC Weights')
    carla_config.add_argument('--actor-weights', default='', type=str, help='Actor pretrained weights path.')
    carla_config.add_argument('--critic-weights', default='', type=str, help='Critic pretrained weights path.')

    # SAC parameters
    rl_group = parser.add_argument_group('RL Config')
    rl_group.add_argument('--num-seed', default=2000, type=int, help='Number of seed steps before starting to train.')
    rl_group.add_argument('--control-frequency', default=4, type=int, help='Number of times that a control signal'
                                                                           'is going to be repeated to the environment')
    rl_group.add_argument('--max-episode-steps', default=200, type=int, help='Maximum number of steps per episode.')
    rl_group.add_argument('--num-eval-episodes', default=3, type=int, help='Number of evaluation episodes.')
    rl_group.add_argument('--num-train-steps', default=1e6, type=int, help='Number of training steps.')
    rl_group.add_argument('--eval-frequency', default=10, type=int, help='number of episodes between evaluations.')
    rl_group.add_argument('--learn-temperature', action='store_true', help='Whether to lean alpha value or not.')
    rl_group.add_argument('--reward-scale', default=1, type=float, help='Reward scale factor (positive)')
    rl_group.add_argument('--speed-reward-weight', default=1, type=float, help='Speed reward weight.')
    rl_group.add_argument('--collision-reward-weight', default=1, type=float, help='Collision reward weight')
    rl_group.add_argument('--lane-distance-reward-weight', default=1, type=float, help='Lane distance reward weight')

    models_parameters = parser.add_argument_group('Actor-Critic config')
    models_parameters.add_argument('--actor-hidden-dim', type=int, default=128, help='Size of hidden layer in the '
                                                                                     'actor model.')
    models_parameters.add_argument('--critic-hidden-dim', type=int, default=128, help='Size of hidden layer in the '
                                                                                      'critic model.')

    loss_parameters = parser.add_argument_group('Loss parameters')
    loss_parameters.add_argument('--actor-l2', type=float, default=4e-2,
                                 help='L2 regularization for the actor model.')
    loss_parameters.add_argument('--critic-l2', type=float, default=4e-2,
                                 help='L2 regularization for the critic model.')

    buffer_group = parser.add_argument_group('Buffer config')
    buffer_group.add_argument('--batch-size', default=1024, type=int, help='Batch size.')
    buffer_group.add_argument('--bc-proportion', default=0.25, type=float,
                              help='Batch size proportion of offline samples that are going to be used for '
                                   'behavioral cloning.')
    buffer_group.add_argument('--online-memory-size', default=8192, type=int, help='Number of steps to be stored in the'
                                                                                   'online buffer')

    # in case of using behavioral cloning
    bc_group = parser.add_argument_group('Behavioral cloning config')
    bc_group.add_argument('--bc', default=None, type=str, help='path to dataset (without extensions)')
    bc_group.add_argument('--debug', action='store_true', help='Whether or not visualize actor input')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # region: GENERAL PARAMETERS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)  # speed up training
    torch.autograd.profiler.profile(False)  # speed up training
    torch.autograd.profiler.emit_nvtx(False)  # speed up training

    control_action_dim = 2
    input_action_dim = 3
    # reward weights for speed, collision and lane distance
    reward_weights = (args.speed_reward_weight, args.collision_reward_weight, args.lane_distance_reward_weight)
    offline_dataset_path = args.bc
    # endregion

    # region: init wandb
    wandb.init(project='tsad', entity='autonomous-driving')
    # endregion

    # region: init env
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
        'dt': 1 / 30,  # time interval between two frames
        'reward_weights': reward_weights,  # reward weights [speed, collision, lane distance]
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration-throttle range
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
    carla_processed_env = EncodeWrapper(carla_raw_env, max_steps=args.max_episode_steps,
                                        reward_scale=args.reward_scale,
                                        action_frequency=args.control_frequency, debug=args.debug)
    carla_processed_env.reset()
    print(colored(f"[+] Environment ready (max_steps={args.max_episode_steps}, reward_scale={args.reward_scale},"
                  f"action_frequency={args.control_frequency})!", "green"))
    # endregion

    # region: init agent
    print(colored("[*] Initializing actor critic models", "white"))
    actor = DiagGaussianActor(action_dim=control_action_dim,
                              hidden_dim=args.actor_hidden_dim,
                              log_std_bounds=(-2, 5))
    actor.load_state_dict(torch.load(args.actor_weights))
    critic = DoubleQCritic(action_dim=input_action_dim,
                           hidden_dim=args.critic_hidden_dim)
    target_critic = DoubleQCritic(action_dim=input_action_dim,
                                  hidden_dim=args.critic_hidden_dim)
    target_critic.load_state_dict(torch.load(args.critic_weights))
    agent = SACAgent(actor=actor,
                     critic=critic,
                     target_critic=target_critic,
                     action_dim=control_action_dim,
                     batch_size=args.batch_size,
                     offline_proportion=args.bc_proportion,
                     actor_weight_decay=args.actor_l2,
                     critic_weight_decay=args.critic_l2,
                     learnable_temperature=args.learn_temperature)
    print(colored("[*] SAC Agent is ready!", "green"))
    # endregion

    # region: init buffer
    print(colored(f"[*] Initializing Mixed Replay Buffer with a size of {args.online_memory_size}", "white"))
    if offline_dataset_path:
        print(colored("BC + RL mode"))
        mixed_replay_buffer = MixedReplayBuffer(args.online_memory_size,
                                                reward_weights=reward_weights,
                                                offline_buffer_hdf5=str(offline_dataset_path) + '.hdf5',
                                                offline_buffer_json=str(offline_dataset_path) + '.json')
    else:
        print(colored("Full DRL mode"))
        mixed_replay_buffer = MixedReplayBuffer(args.online_memory_size,
                                                reward_weights=reward_weights)
    print(colored("[*] The replay Buffer is ready!", "green"))
    # endregion

    train_params = {
        "device": "cuda",
        "seed": 42,
        "log_save_tb": True,
        "num_eval_episodes": args.num_eval_episodes,  # number of episodes used for evaluation
        "num_train_steps": args.num_train_steps,  # number of training steps
        "eval_frequency": args.eval_frequency,  # number of steps required for evaluation
        "num_seed_steps": args.num_seed  # number of steps before starting to update the models
    }
    print(colored("Evaluating", "white"))
    trainer = SACTrainer(env=carla_processed_env,
                         agent=agent,
                         buffer=mixed_replay_buffer,
                         log_eval=True,
                         **train_params)
    try:
        trainer.evaluate()
    except Exception as e:
        print(colored("\nEarly stopping due to exception", "red"))
        traceback.print_exc()
        print(e)
    finally:
        print(colored("\nTraning finished!", "green"))
        trainer.end()
