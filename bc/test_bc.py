import argparse

import numpy as np
import torch
import wandb
from gym_carla.envs.carla_pid_env import CarlaPidEnv
from gym_carla.envs.carla_env import CarlaEnv
from torch.utils.data import DataLoader

from train_bc import BCTrainer

from models.carlaAffordancesDataset import AffordancesDataset, HLCAffordanceDataset
from bc_agent import BCStochasticAgent, BCDeterministicAgent



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Settings for the data capture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-H', '--host', default='localhost', type=str, help='CARLA server ip address')
    parser.add_argument('-p', '--port', default=2000, type=int, help='CARLA server port number')
    parser.add_argument('--tm-port', default=8000, type=int, help='Traffic manager port')
    parser.add_argument('-t', '--town', default='Town01', type=str, help="town to use")
    parser.add_argument('-ve', '--vehicles', default=100, type=int,
                        help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=0, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('--act-mode', default="raw", type=str, help="Action space. 'raw': raw actions (throttle, brake,"
                                                                    "steer), 'pid': (target_speed, steer)")
    parser.add_argument('--eval-frequency', default=50, type=int)
    parser.add_argument('--open-loop-eval-frequency', default=20, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--checkpoint', default='checkpoint.pt', type=str)
    parser.add_argument('--next-speed', action='store_true', help='Whether to use next step speed head or not.')
    parser.add_argument('--norm-actions', action='store_true', help='Whether the action are normalized or not.')

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='tsad', entity='autonomous-driving', name='bc-test')

    params = {
        # carla connection parameters+
        'host': args.host,
        'port': args.port,  # connection port
        'town': 'Town01',  # which town to simulate
        'traffic_manager_port': args.tm_port,

        # simulation parameters
        'verbose': False,
        'vehicles': args.vehicles,  # number of vehicles in the simulation
        'walkers': args.walkers,  # number of walkers in the simulation
        'obs_size': 224,  # sensor width and height
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.025,  # time interval between two frames
        'reward_weights': [0.3, 0.3, 0.3],

        'normalized_input': args.norm_actions,
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'max_time_episode': 500,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'speed_reduction_at_intersection': 0.75,
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    }
    env = None
    if args.eval_frequency > 0:
        if args.act_mode == "pid":
            params.update({
                'continuous_speed_range': [0.0, 6.0],
                'continuous_steer_range': [-1.0, 1.0],
            })
            env = CarlaPidEnv(params)
        else:
            params.update({
                'continuous_throttle_range': [0.0, 1.0],
                'continuous_brake_range': [0.0, 1.0],
                'continuous_steer_range': [-1.0, 1.0],
            })
            env = CarlaEnv(params)
    action_dim = 2 if args.act_mode == "pid" else 3

    agent = BCStochasticAgent(input_size=15, hidden_dim=512, action_dim=action_dim, log_std_bounds=(-2, 5),
                              checkpoint=args.checkpoint, next_speed=args.next_speed)
    agent.load()
    # dataset = AffordancesDataset(args.data)
    trainer = BCTrainer(agent, None, env,
                        use_wandb=args.wandb,
                        eval_episodes=args.epochs,
                        eval_frequency=args.eval_frequency,
                        open_loop_eval_frequency=args.open_loop_eval_frequency,
                        use_next_speed=args.next_speed,
                        norm_actions=args.norm_actions,
                        )
    
    trainer.eval()
