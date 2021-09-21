import argparse
import traceback
import warnings

import torch
import wandb
from gym_carla.envs.carla_env import CarlaEnv
from gym_carla.envs.carla_pid_env import CarlaPidEnv
from termcolor import colored
from torch.utils.data import DataLoader

from bc.train_bc import get_collate_fn
from models.carlaAffordancesDataset import HLCAffordanceDataset, AffordancesDataset
from sac.replay_buffer import OnlineReplayBuffer
from sac.sac_agent import SACAgent
from sac.trainer import SACTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAC Trainer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # carla parameters
    carla_config = parser.add_argument_group('CARLA config')
    carla_config.add_argument('--host', default='172.18.0.1', type=str, help='IP address of CARLA host.')
    carla_config.add_argument('--port', default=2008, type=int, help='Port number of CARLA host.')
    carla_config.add_argument('--vehicles', default=100, type=int, help='Number of vehicles in the simulation.')
    carla_config.add_argument('--walkers', default=50, type=int, help='Number of walkers in the simulation.')

    # SAC parameters
    rl_group = parser.add_argument_group('RL Config')
    rl_group.add_argument('--num-seed', default=2000, type=int, help='Number of seed steps before starting to train.')
    rl_group.add_argument('--control-frequency', default=4, type=int, help='Number of times that a control signal'
                                                                           'is going to be repeated to the environment')
    rl_group.add_argument('--act-mode', default="pid", type=str, help="Action space.")
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
    models_parameters.add_argument('--actor-weights', type=str, default=None, help='Path to actor weights')
    models_parameters.add_argument('--critic-weights', type=str, default=None, help='Path to critic weights')

    loss_parameters = parser.add_argument_group('Loss parameters')
    loss_parameters.add_argument('--actor-l2', type=float, default=4e-2,
                                 help='L2 regularization for the actor model.')
    loss_parameters.add_argument('--critic-l2', type=float, default=4e-2,
                                 help='L2 regularization for the critic model.')

    buffer_group = parser.add_argument_group('Buffer config')
    buffer_group.add_argument('--batch-size', default=1024, type=int, help='Batch size.')
    buffer_group.add_argument('--online-memory-size', default=8192, type=int, help='Number of steps to be stored in the'
                                                                                   'online buffer')

    # in case of using behavioral cloning
    bc_group = parser.add_argument_group('Behavioral cloning config')
    bc_group.add_argument('--bc', default=None, type=str, help='path to dataset (without extensions)')
    bc_group.add_argument('--wandb', action='store_true', help='Whether or not to use wandb')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    control_action_dim = 2 if args.act_mode == "pid" else 3
    action_range = (-1, 1) if args.act_mode == "raw" else (-1, 5)
    offline_dataset_path = args.bc

    if args.wandb:
        wandb.init(project='tsad', entity='autonomous-driving')

    carla_env = None
    if args.eval_frequency > 0:
        print(colored("[*] Initializing environment", "white"))
        desired_speed = 6
        env_params = {
            # carla connection parameters+
            'host': args.host,
            'port': args.port,  # connection port
            'town': 'Town01',  # which town to simulate
            'traffic_manager_port': 8000,

            # simulation parameters
            'verbose': False,
            'vehicles': args.vehicles,  # number of vehicles in the simulation
            'walkers': args.walkers,  # number of walkers in the simulation
            'obs_size': 224,  # sensor width and height
            'max_past_step': 1,  # the number of past steps to draw
            'dt': 1 / 30,  # time interval between two frames
            'reward_weights': [1, 1, 1],  # reward weights [speed, collision, lane distance]
            'continuous_steer_range': [-1, 1],
            'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
            'max_time_episode': args.max_episode_steps,  # maximum timesteps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'd_behind': 12,  # distance behind the ego vehicle (meter)
            'out_lane_thres': 2.0,  # threshold for out of lane
            'desired_speed': desired_speed,  # desired speed (m/s)
            'speed_reduction_at_intersection': 0.75,
            'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        }

        if args.act_mode == "pid":
            env_params.update({
                'continuous_speed_range': [0, desired_speed]
            })
            carla_env = CarlaPidEnv(env_params)
        else:
            env_params.update({
                'continuous_throttle_range': [0, 1],
                'continuous_brake_range': [0, 1]
            })
            carla_env = CarlaEnv(env_params)
        carla_env.reset()
        print(colored(f"[+] Environment ready "
                      f"(max_steps={args.max_episode_steps},"
                      f"action_frequency={args.control_frequency})!", "green"))

    print(colored(f"[*] Initializing data structures", "white"))
    online_replay_buffer = OnlineReplayBuffer(args.online_memory_size)
    bc_loaders = None
    if offline_dataset_path:
        print(colored("RL + BC mode"))
        dataset = AffordancesDataset(args.bc)
        custom_collate_fn = get_collate_fn(args.act_mode)
        bc_loaders = {hlc: DataLoader(HLCAffordanceDataset(dataset, hlc=hlc),
                                      batch_size=args.batch_size,
                                      collate_fn=custom_collate_fn,
                                      shuffle=True) for hlc in [0, 1, 2, 3]}
    else:
        print(colored("Full DRL mode"))
    print(colored("[*] Data structures are ready!", "green"))

    agent = SACAgent(observation_dim=15,
                     action_range=action_range,
                     device=device,
                     action_dim=control_action_dim,
                     batch_size=args.batch_size,
                     actor_weight_decay=args.actor_l2,
                     critic_weight_decay=args.critic_l2,
                     learnable_temperature=args.learn_temperature)
    agent.train(True)

    print(colored("Training", "white"))
    trainer = SACTrainer(env=carla_env,
                         agent=agent,
                         buffer=online_replay_buffer,
                         dataloaders=bc_loaders,
                         device=device,
                         eval_frequency=args.eval_frequency,
                         num_seed_steps=args.num_seed,
                         num_train_steps=args.num_train_steps,
                         num_eval_episodes=args.num_eval_episodes,
                         seed=42)

    try:
        trainer.run()
    except Exception as e:
        print(colored("\nEarly stopping due to exception", "red"))
        traceback.print_exc()
        print(e)
    finally:
        print(colored("\nTraning finished!", "green"))
        trainer.end()
