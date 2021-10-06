import argparse
import sys
import time

import numpy as np
import torch
import wandb
from typing import Union
from gym_carla.envs.carla_pid_env import CarlaPidEnv
from gym_carla.envs.carla_env import CarlaEnv
from torch.utils.data import DataLoader
from bc.utils import normalize_action, normalize_pid_action

from agents.agent import MultiTaskAgent
from bc_agent import BCStochasticAgent, BCDeterministicAgent
from models.carlaAffordancesDataset import AffordancesDataset, HLCAffordanceDataset

V_MAX = 15
HLC_TO_NUMBER = {
    'RIGHT': 0,
    'LEFT': 1,
    'STRAIGHT': 2,
    'LANEFOLLOW': 3
}


def get_number_order(num):
    order = 0
    while num != 0:
        num = num // 10
        order += 1
    return order


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def get_collate_fn(act_mode: str):
    def collate_fn(samples: list) -> (dict, dict):
        """
        Returns a dictionary with grouped samples. Each sample is a tuple which comes from the dataset __getitem__.
        """
        obs, act = [], []
        for t in samples:
            obs.append(dict(encoding=t[0]))
            if act_mode == "pid":
                norm_speed = 2 * t[2] / V_MAX - 1
                act.append(np.array([norm_speed, t[1][2]]))  # t[2] = speed, t[1] = control (throttle, brake, steer)
            else:
                act.append(t[1])
        return obs, act
    return collate_fn


def get_normalizer(act_mode: str):
    if act_mode == "pid":
        return lambda x: np.array(normalize_action(x))
    elif act_mode == "raw":
        return lambda x: np.array(normalize_pid_action(x))
    else:
        raise ValueError("Act mode not allowed.")


class BCTrainer(object):
    """
    Auxiliary class to train a policy using Behavioral Cloning over affordances trajectories.
    """

    def __init__(self,
                 actor: Union[BCStochasticAgent, BCDeterministicAgent],
                 dataset: AffordancesDataset,
                 env: Union[CarlaPidEnv, CarlaEnv],
                 action_space: str = "pid",
                 batch_size: int = 128,
                 epochs: int = 100,
                 eval_frequency: int = 10,
                 open_loop_eval_frequency: int = 2,
                 eval_episodes: int = 3,
                 use_wandb: bool = False,
                 use_next_speed: bool = False,
                 ):
        self._actor = actor
        self._dataset = dataset
        self._epochs = epochs
        self._env = env
        self._wandb = use_wandb
        self._eval_frequency = eval_frequency
        self._open_loop_eval_frequency = open_loop_eval_frequency
        self._nb_eval_episode = eval_episodes
        self._eval_step = 0
        self._eval_episode = 0
        self._use_next_speed = use_next_speed

        self._batch_size = batch_size
        self.__hlc_to_train = [0, 1, 2, 3]

        if dataset:
            act_collate_fn = get_collate_fn(action_space)
            normalizer = get_normalizer(action_space)
            self._train_loaders = {hlc: DataLoader(HLCAffordanceDataset(self._dataset,
                                                                        hlc=hlc,
                                                                        use_next_speed=use_next_speed,
                                                                        normalizer=normalizer),
                                                   batch_size=self._batch_size, collate_fn=act_collate_fn,
                                                   shuffle=True) for hlc in self.__hlc_to_train}
        self.mse = torch.nn.MSELoss()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def eval(self):
        print("\nEvaluating...")
        total_reward = 0
        self._actor.eval_mode()
        for e in range(self._nb_eval_episode):
            obs = self._env.reset()
            episode_reward = 0
            done = False
            while not done:
                start = time.time()
                action = self._actor.act_single(obs, task=obs["hlc"] - 1)
                action[0] = (V_MAX / 2) * (action[0] + 1)   # denormalize the speed
                speed = np.linalg.norm(obs["speed"])
                obs, rew, done, _ = self._env.step(action=action)
                episode_reward += rew

                fps = 1 / (time.time() - start)
                action_str = ("{:.2f}, "*len(action)).format(*action)
                sys.stdout.write("\r")
                sys.stdout.write(f"fps={fps:.2f} action={action_str} speed={speed:.2f} rew={rew:.2f}")
                sys.stdout.flush()

                if self._wandb:
                    wandb.log({
                        "eval/instant/speed": speed,
                        "eval/instant/target_speed": action[0],
                        "eval/instant/steer": action[1],
                        "eval/step": self._eval_step,
                    })
                self._eval_step += 1

            if self._wandb:
                wandb.log({
                    "eval/episode_reward": episode_reward,
                    "eval/episode": self._eval_episode,
                })
            self._eval_episode += 1
            print(f"\nEval episode {e}: {episode_reward:.2f}")
            total_reward += episode_reward

        avg_reward = total_reward / self._nb_eval_episode
        print(f"Avg reward: {avg_reward:.2f}")
        if self._wandb:
            wandb.log({
                "eval_actor/eval_reward": avg_reward})
        self._actor.train_mode()

    def open_loop_eval(self):
        """
        Performs open-loop evaluation in the given dataset.
        """
        self._actor.eval_mode()
        print("\n" + "-" * 50)
        print("Open loop evaluation:")
        for hlc in self.__hlc_to_train:
            hlc_loader = self._train_loaders[hlc]
            hlc_loss = 0
            for i, (obs, act) in enumerate(hlc_loader):
                act_expert = torch.tensor(np.stack(act), device=self._device).float()
                act_pred = self._actor.act_batch(obs, hlc)
                hlc_loss += self.mse(act_pred, act_expert)
            hlc_loss /= len(hlc_loader)
            print(f"HLC {hlc} MSE Loss: {hlc_loss}")
            if self._wandb:
                wandb.log({f"open_loop_eval/mse_{hlc}": hlc_loss})
        print("-" * 50)
        self._actor.train_mode()

    def run(self):
        if self._wandb:
            wandb.init(project='tsad', entity='autonomous-driving')

        steps = {0: 0, 1: 0, 2: 0, 3: 0}
        for e in range(1, self._epochs + 1):
            for hlc in self.__hlc_to_train:
                hlc_loader = self._train_loaders[hlc]

                for i, (obs, act) in enumerate(hlc_loader):
                    # parse to torch tensors
                    obs = torch.stack([torch.tensor(o['encoding'], device=self._device) for o in obs], dim=0).float()
                    act = torch.tensor(act, device=self._device).float()

                    # update using bc and get the loss
                    bc_loss = self._actor.get_supervised_loss(obs, act, hlc)

                    if self._wandb:
                        wandb.log({f'train_actor/bc_loss_{hlc}': bc_loss,
                                   f'steps_{hlc}': steps[hlc]})
                    steps[hlc] += 1

                    sys.stdout.write("\r")
                    sys.stdout.write(
                        f"Epoch={str(e).zfill(get_number_order(self._epochs))}(hlc={hlc}) "
                        f"{str(i).zfill(get_number_order(len(hlc_loader)))}/{len(hlc_loader)}] bc_loss={bc_loss:.5f}")
                    sys.stdout.flush()
            print("")
            if self._env and e % self._eval_frequency == 0:
                self.eval()
                self._actor.save()
            if e % self._open_loop_eval_frequency == 0:
                self.open_loop_eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Settings for the data capture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to data folder')
    parser.add_argument('-H', '--host', default='localhost', type=str, help='CARLA server ip address')
    parser.add_argument('-p', '--port', default=2000, type=int, help='CARLA server port number')
    parser.add_argument('--tm-port', default=8000, type=int, help='Traffic manager port')
    parser.add_argument('-t', '--town', default='Town01', type=str, help="town to use")
    parser.add_argument('-ve', '--vehicles', default=100, type=int,
                        help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=0, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('--act-mode', default="raw", type=str, help="Action space. 'raw': raw actions (throttle, brake,"
                                                                    "steer), 'pid': (target_speed, steer)")
    parser.add_argument('--eval-frequency', default=50, type=int, help="Closed-loop evaluation frequency."
                                                                       "If -1, it is omitted.")
    parser.add_argument('--open-loop-eval-frequency', default=20, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--checkpoint', default='checkpoint.pt', type=str)
    parser.add_argument('--next-speed', action='store_true', help='Whether to use next step speed head or not.')

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='tsad', entity='autonomous-driving', name='bc-train')

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
    dataset = AffordancesDataset(args.data)
    trainer = BCTrainer(agent, dataset, env,
                        action_space=args.act_mode,
                        use_wandb=args.wandb,
                        epochs=args.epochs,
                        eval_frequency=args.eval_frequency,
                        open_loop_eval_frequency=args.open_loop_eval_frequency,
                        use_next_speed=args.next_speed
                        )
    trainer.run()
