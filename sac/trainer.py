import os
import sys
import time
import numpy as np
import wandb

import torch

import sac.utils as utils
from sac.rl_logger import RLLogger
from typing import Dict, Union
from torch.utils.data import DataLoader
from sac.replay_buffer import OnlineReplayBuffer
from gym_carla.envs.carla_env import CarlaEnv
from gym_carla.envs.carla_pid_env import CarlaPidEnv
from sac.sac_agent import SACAgent


ROAD_OPTION_TO_NAME = {
    0: "Left",
    1: "Right",
    2: "Straight",
    3: "Lane Follow"
}


def action_proxy(act: np.ndarray) -> list:
    """
    Parse an action from two dimensions to three dimensions
    """
    acc = act[0]
    steer = act[1]
    # Convert acceleration to throttle and brake
    if acc > 0:
        throttle = np.clip(acc, 0, 1)
        brake = 0
    else:
        throttle = 0
        brake = np.clip(-acc, 0, 1)
    return [throttle, brake, steer]


class SACTrainer(object):
    def __init__(self,
                 env: Union[CarlaEnv, CarlaPidEnv],
                 agent: SACAgent,
                 buffer: OnlineReplayBuffer,
                 dataloaders: Dict[str, DataLoader],
                 log_eval=False,
                 wandb: bool = False,
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
        self.bc_loaders = dataloaders

        self.wandb = wandb
        self.env = env
        self.agent = agent
        self.replay_buffer = buffer
        self.logger = RLLogger()
        self.best_average_episode_reward = -1e6
        self.step = 0

        self.log_eval = log_eval

    def evaluate(self):
        average_episode_reward = 0
        steps = 0
        with utils.eval_mode(self.agent):
            for episode in range(self.num_eval_episodes):
                duration = 0
                obs = self.env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = self.agent.act(obs, sample=False)
                    obs, reward, done, _ = self.env.step(action)
                    if self.log_eval and self.wandb:
                        wandb.log({f"instant/action/action_{i}": value for i, value in enumerate(action)})
                        wandb.log({"instant/reward": reward})
                        wandb.log({"eval/step": steps})
                    episode_reward += reward
                    steps += 1
                    duration += 1

            if self.log_eval and self.wandb:
                wandb.log({'eval/episode': episode, 'eval/episode_reward': episode_reward, 'eval/duration': duration})
            average_episode_reward += episode_reward
        average_episode_reward /= self.num_eval_episodes

        if average_episode_reward > self.best_average_episode_reward:
            print(f"Saving Actor-Critic at {self.work_dir} (step={self.step})")
            self.best_average_episode_reward = average_episode_reward
            self.agent.save(self.work_dir)

        if self.wandb:
            wandb.log({'eval/episode_reward': average_episode_reward})

    def run(self):
        episode, episode_reward, done, episode_step = 0, 0, 0, True
        obs = self.env.reset()
        start_time = time.time()
        self.agent.train(True)
        while self.step < self.num_train_steps:
            if done:
                if self.step > 0 and self.wandb:
                    wandb.log({'train/duration': time.time() - start_time})
                    start_time = time.time()

                # evaluate agent periodically
                if self.step > 0 and episode % self.eval_frequency == 0:
                    print("\nEvaluating...")
                    self.evaluate()
                    print("Done!")

                if self.wandb:
                    wandb.log({'train/episode_reward': episode_reward,
                               'train/episode': episode})

                print("\nResetting")
                obs = self.env.reset()
                episode_reward = 0
                episode_step = 0
                episode += 1

            # sample action for data collection
            if self.step < self.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            if self.wandb:
                wandb.log({f"instant/action/{name}": value for name, value in zip(["throttle", "brake", "steer"],
                                                                                  action)})
                wandb.log({"instant/speed": np.linalg.norm(obs["speed"])})

            # run training update
            if self.step >= self.num_seed_steps:
                self.agent.update(self.replay_buffer, self.step)

            # save checkpoints
            if self.step % 50000 == 0:
                self.agent.save(self.work_dir, tag=f"at_{self.step}")

            next_obs, reward, done, _ = self.env.step(action)
            not_done = 1 - float(done)
            episode_reward += reward

            if self.wandb:
                wandb.log({"instant/reward": reward,
                           "rl_step": self.step})

            self.replay_buffer.add(obs, action, reward, next_obs, not_done)

            obs = next_obs
            episode_step += 1
            self.step += 1

            self.logger.log(f"[{'seed' if self.step < self.num_seed_steps else 'train'}]"
                            f"({(self.step if self.step <= 1000 else self.step/1000):.2f}"
                            f"{'k' if self.step > 1000 else ''}/"
                            f"{self.num_train_steps/1000:.0f}k), "
                            f"[{episode}:{episode_step}:{ROAD_OPTION_TO_NAME[obs['hlc']]}]| "
                            f"rew={reward:.2f}")

            sys.stdout.write("\r")
            sys.stdout.write(f"Training step: {self.step}/{self.num_train_steps}"
                             f"{self.replay_buffer.log_status() if self.step % 2000 == 0 else ''}")

    def end(self):
        # save last agent
        self.agent.save(self.work_dir, tag="last")
        self.env.reset()
        self.logger.close()
