#!/usr/bin/env python3
import os
import sys
import time

import cv2
import torch

import sac.utils as utils
from models.carla_wrapper import EncodeWrapper
from replay_buffer import MixedReplayBuffer
from sac.agent.sac import SACAgent
from termcolor import colored
from sac.rl_logger import RLLogger

ROAD_OPTION_TO_NAME = {
    0: "Left",
    1: "Right",
    2: "Straight",
    3: "Lane Follow"
}


class SACTrainer(object):
    def __init__(self,
                 env: EncodeWrapper,
                 agent: SACAgent,
                 buffer: MixedReplayBuffer,
                 **kwargs):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        # save some training configurations
        utils.set_seed_everywhere(kwargs["seed"])
        self.device = torch.device(kwargs["device"])
        self.num_eval_episodes = kwargs["num_eval_episodes"]
        self.num_train_steps = kwargs["num_train_steps"]
        self.num_seed_steps = kwargs["num_seed_steps"]
        self.eval_frequency = kwargs["eval_frequency"]

        self.env = env
        self.agent = agent
        self.replay_buffer = buffer
        self.logger = RLLogger()

        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward

            average_episode_reward += episode_reward
        average_episode_reward /= self.num_eval_episodes
        wandb.log({'eval/episode_reward': average_episode_reward})

    def run(self):
        episode, episode_reward, done, episode_step = 0, 0, 0, True
        obs = self.env.reset()
        start_time = time.time()
        while self.step < self.num_train_steps:
            if done:
                if self.step > 0:
                    wandb.log({'train/duration': time.time() - start_time})
                    start_time = time.time()

                # evaluate agent periodically
                if self.step > 0 and self.step % self.eval_frequency == 0:
                    wandb.log({'eval/episode': episode})
                    self.evaluate()

                wandb.log({'train/episode_reward': episode_reward})

                print("\nResetting")
                obs = self.env.reset()
                self.agent.reset()
                episode_reward = 0
                episode_step = 0
                episode += 1
                wandb.log({'train/episode': episode})

            # sample action for data collection
            if self.step < self.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.num_seed_steps:
                self.agent.update(self.replay_buffer, self.step)

            next_obs, reward, done, _ = self.env.step(action)
            not_done = 1 - float(done)
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, not_done)

            obs = next_obs
            episode_step += 1
            self.step += 1

            self.logger.log(f"({self.step}/{self.num_train_steps:.0f}), "
                            f"[{episode}:{episode_step}:{ROAD_OPTION_TO_NAME[obs['hlc']]}]| "
                            f"reward: {reward:.2f}, acc: {action[0]:.3f}, steer: {action[1]:.3f}, done: {done}")

            sys.stdout.write("\r")
            sys.stdout.write(f"Training step: {self.step}/{self.num_train_steps}")

    def end(self):
        self.env.reset()
        cv2.destroyAllWindows()
        self.logger.close()


if __name__ == '__main__':
    import warnings
    import wandb
    from gym_carla.envs.carla_env import CarlaEnv
    from models.ADEncoder import ADEncoder
    from models.TemporalEncoder import VanillaRNNEncoder
    from sac.agent.sac import SACAgent
    from sac.agent.actor import DiagGaussianActor
    from sac.agent.critic import DoubleQCritic
    import argparse

    parser = argparse.ArgumentParser(description="SAC Trainer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, default=None, type=str, help='path to hdf5 file')
    parser.add_argument('-m', '--metadata', default=None, type=str, help='path to json file')
    parser.add_argument('--debug', action='store_true', help='Whether or not visualize actor input')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # region: GENERAL PARAMETERS
    action_dim = 2
    actor_hidden_dim = 512
    critic_hidden_dim = 512

    online_memory_size = 512
    offline_dataset_path = {
        "hdf5": args.input,
        "json": args.metadata
    }
    # endregion

    # region: init wandb
    wandb.init(project='tsad', entity='autonomous-driving')
    # endregion

    # region: init env
    print(colored("[*] Initializing models", "white"))
    visual = ADEncoder(backbone='efficientnet-b5')
    temp = VanillaRNNEncoder()
    visual.freeze()
    temp.freeze()
    print(colored("[+] Encoder models were initialized and loaded successfully!", "green"))

    print(colored("[*] Initializing environment", "white"))
    env_params = {
        'number_of_vehicles': 0,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.05,  # time interval between two frames
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town01',  # which town to simulate
        'max_time_episode': 100,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'reduction_at_intersection': 0.75
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
    mixed_replay_buffer = MixedReplayBuffer(online_memory_size)
    # offline_buffer_hdf5=offline_dataset_path["hdf5"],
    # offline_buffer_json=offline_dataset_path["json"])
    print(colored("[*] Initializing Mixed Replay Buffer", "green"))
    # endregion

    train_params = {
        "device": "cuda",
        "seed": 42,
        "log_save_tb": True,
        "num_eval_episodes": 5,
        "num_train_steps": 1e6,
        "eval_frequency": 10000,
        "num_seed_steps": 200
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
