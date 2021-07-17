import os
import sys
import time
import numpy as np
import wandb

import cv2
import torch

import sac.utils as utils
from models.carla_wrapper import EncodeWrapper
from replay_buffer import MixedReplayBuffer
from sac.agent.sac import SACAgent
from sac.rl_logger import RLLogger

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
        self.best_average_episode_reward = -1e6
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
                obs, reward, done, _ = self.env.step(action_proxy(action))
                episode_reward += reward

            average_episode_reward += episode_reward
        average_episode_reward /= self.num_eval_episodes

        if average_episode_reward > self.best_average_episode_reward:
            print(f"Saving Actor-Critic at {self.work_dir} (step={self.step})")
            self.best_average_episode_reward = average_episode_reward
            self.agent.save(self.work_dir)

        wandb.log({'eval/episode_reward': average_episode_reward})

    def run(self):
        episode, episode_reward, done, episode_step = 0, 0, 0, True
        obs = self.env.reset()
        start_time = time.time()
        self.agent.train()
        while self.step < self.num_train_steps:
            if done:
                if self.step > 0:
                    wandb.log({'train/duration': time.time() - start_time})
                    start_time = time.time()

                # evaluate agent periodically
                if self.step > 0 and episode % self.eval_frequency == 0:
                    wandb.log({'eval/episode': episode})
                    print("\nEvaluating...")
                    self.evaluate()
                    print("Done!")

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

            # AGENT ACTION DIM PROXY: 2 -> 3
            action = action_proxy(action)

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

            self.logger.log(f"[{'seed' if self.step < self.num_seed_steps else 'train'}]"
                            f"({(self.step if self.step <= 1000 else self.step/1000):.2f}"
                            f"{'k' if self.step > 1000 else ''}/"
                            f"{self.num_train_steps/1000:.0f}k), "
                            f"[{episode}:{episode_step}:{ROAD_OPTION_TO_NAME[obs['hlc']]}]| "
                            f"rew={reward:.2f}, acc={action[0]:.3f}, brake={action[1]:.3f} "
                            f"steer={action[2]:.3f}, done={done}")

            sys.stdout.write("\r")
            sys.stdout.write(f"Training step: {self.step}/{self.num_train_steps}")
            sys.stdout.write(self.replay_buffer.log_status() if self.step % 2000 == 0 else "")
            sys.stdout.flush()

    def end(self):
        self.env.reset()
        cv2.destroyAllWindows()
        self.logger.close()
