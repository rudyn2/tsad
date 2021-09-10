import torch
import wandb
import sys
import argparse
from torch.utils.data import DataLoader
import numpy as np
import time

from sac.agent.actor import DiagGaussianActor
from models.carlaAffordancesDataset import AffordancesDataset, HLCAffordanceDataset
from gym_carla.envs.carla_env import CarlaEnv

HLC_TO_NUMBER = {
    'RIGHT': 0,
    'LEFT': 1,
    'STRAIGHT': 2,
    'LANEFOLLOW': 3
}


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def collate_fn(samples: list) -> (dict, dict):
    """
    Returns a dictionary with grouped samples. Each sample is a tuple which comes from the dataset __getitem__.
    """
    obs, act = [], []
    for t in samples:
        obs.append(dict(encoding=t[0]))
        act.append(t[1])
    return obs, act


class BCTrainer(object):
    """
    Auxiliary class to train a policy using Behavioral Cloning over affordances trajectories.
    """

    def __init__(self,
                 actor: DiagGaussianActor,
                 dataset: AffordancesDataset,
                 env: CarlaEnv,
                 lr: float = 0.0001,
                 batch_size: int = 128,
                 epochs: int = 100,
                 eval_frequency: int = 10,
                 eval_episodes: int = 3,
                 use_wandb: bool = False
                 ):
        self._actor = actor
        self._dataset = dataset
        self._epochs = epochs
        self._env = env
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=lr)
        self._wandb = use_wandb
        self._eval_frequency = eval_frequency
        self._eval_episode = eval_episodes

        self._batch_size = batch_size
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._actor.to(self._device)

    def eval(self):
        print("\nEvaluating...")
        total_reward = 0
        for e in range(self._eval_episode):
            obs = self._env.reset()
            episode_reward = 0
            done = False
            while not done:
                start = time.time()
                dist = self._actor(dict(encoding=obs["affordances"]), hlc=int(obs["hlc"])-1)
                action = dist.mean
                action = action.clamp(-1, 1)
                action = list(to_np(action[0]))
                speed = np.linalg.norm(obs["speed"])
                obs, rew, done, _ = self._env.step(action=action)
                episode_reward += rew

                fps = 1 / (time.time() - start)
                sys.stdout.write("\r")
                sys.stdout.write(f"fps={fps:.2f} speed={speed:.2f} rew={rew}")
                sys.stdout.flush()

                if self._wandb:
                    wandb.log({
                        "eval/instant/speed": speed,
                        "eval/instant/throttle": action[0],
                        "eval/instant/brake": action[1],
                        "eval/instant/steer": action[2],
                    })

            if self._wandb:
                wandb.log({
                    "eval/episode_reward": episode_reward
                })
            print(f"\nEval episode {e}: {episode_reward:.2f}")
            total_reward += episode_reward

        avg_reward = total_reward / self._eval_episode
        print(f"Avg reward: {avg_reward:.2f}")
        if self._wandb:
            wandb.log({"eval_actor/eval_reward": avg_reward})

    def run(self):
        if self._wandb:
            wandb.init(project='tsad', entity='autonomous-driving')

        train_loaders = {hlc: DataLoader(HLCAffordanceDataset(self._dataset, hlc=hlc),
                                         batch_size=self._batch_size, collate_fn=collate_fn,
                                         shuffle=True) for hlc in range(4)}
        steps = {0: 0, 1: 0, 2: 0, 3: 0}
        for e in range(1, self._epochs + 1):
            for hlc in range(4):
                hlc_loader = train_loaders[hlc]

                for i, (obs, act) in enumerate(hlc_loader):

                    dist_e = self._actor(obs, hlc=hlc)
                    act_e_hlc = torch.tensor(np.stack(act), device=self._device).float()
                    log_prob_e = dist_e.log_prob(torch.clamp(act_e_hlc, min=-1 + 1e-6, max=1.0 - 1e-6)).sum(-1,
                                                                                                            keepdim=True)
                    bc_loss = - log_prob_e.mean()
                    if self._wandb:
                        wandb.log({f'train_actor/bc_loss_{hlc}': bc_loss.item(),
                                   f'steps_{hlc}': steps[hlc]})
                    steps[hlc] += 1

                    self._actor_optimizer.zero_grad()
                    bc_loss.backward()
                    self._actor_optimizer.step()

                    sys.stdout.write("\r")
                    sys.stdout.write(f"Epoch={e}(hlc={hlc}) [{i}/{len(hlc_loader)}] bc_loss={bc_loss.item():.2f}")
                    sys.stdout.flush()

            if e % self._eval_frequency == 0:
                self.eval()


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
    parser.add_argument('--eval-frequency', default=50, type=int)
    parser.add_argument('--epochs', default=15, type=int)

    args = parser.parse_args()

    wandb.init(project='tsad', entity='autonomous-driving', name='bc-train')

    env = CarlaEnv({
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
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'speed_reduction_at_intersection': 0.75,
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    })
    actor = DiagGaussianActor(input_size=15, hidden_dim=64, action_dim=3, log_std_bounds=(-2, 5))
    dataset = AffordancesDataset(args.data)
    trainer = BCTrainer(actor, dataset, env, use_wandb=False,
                        epochs=args.epochs, eval_frequency=args.eval_frequency)
    trainer.run()
