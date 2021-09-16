import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

import sac.utils as utils
from replay_buffer import OnlineReplayBuffer
from sac_ac import SACActor, SACCritic


def to_tensor(*args, device: str):
    tensors = []
    for arg in args:
        tensors.append(torch.tensor(arg, device=device).float())
    return tensors


# https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac


class SACAgent(object):
    """SAC algorithm."""

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,  # used only for target entropy definition
                 action_range: tuple,
                 device: str = 'cuda',
                 wandb: bool = False,
                 discount: float = 0.99,
                 init_temperature: float = 0.3,
                 critic_tau: float = 0.005,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 alpha_lr: float = 1e-4,
                 actor_weight_decay: float = 4e-2,
                 critic_weight_decay: float = 4e-2,
                 actor_update_frequency: int = 1,
                 critic_target_update_frequency: int = 2,
                 batch_size: int = 1024,
                 actor_bc_update_frequency: int = 4,
                 learnable_temperature: bool = True):
        super().__init__()

        self.device = torch.device(device)
        self.action_range = action_range
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.actor_bc_update_frequency = actor_bc_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self._wandb = wandb

        # store actor and critic
        self.critic = SACCritic(input_dim=observation_dim + action_dim, hidden_dim=(512, 512))
        self.critic_target = SACCritic(input_dim=observation_dim + action_dim, hidden_dim=(512, 512))
        # noinspection PyTypeChecker
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = SACActor(input_dim=observation_dim, hidden_dim=(512, 512), output_dim=action_dim)

        # move actor and critic to specified device
        self.critic.to(self.device)
        self.actor.to(self.device)
        self.critic_target.to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                weight_decay=actor_weight_decay)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 weight_decay=critic_weight_decay)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)

        self.train()
        self.critic_target.train()
        self._step = 0
        self.__hlc_to_train = [3]  # WHEN EVERYTHING IS READY, CHANGE THIS TO ALL THE HIGH LEVEL TASKS

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False

    def train(self):
        self.actor.train_mode()
        self.critic.train_mode()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        dist = self.actor(obs, obs['hlc'])
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return utils.to_np(action[0])

    def update_critic(self, obs, act, reward, next_obs, not_done, hlc: int):

        # unpack and format
        obs = [o["affordances"] for o in obs]
        next_obs = [o["affordances"] for o in next_obs]
        obs_t, act_t, rew_t, next_obs_t, not_done_t = to_tensor(obs, act, reward, next_obs, not_done, device=self.device)

        next_action, log_prob = self.actor(next_obs_t, hlc)
        target_q1, target_q2 = self.critic_target(next_obs_t, next_action, hlc)
        target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * log_prob
        target_q = rew_t + (not_done_t * self.discount * target_v).float()
        target_q = target_q.detach()
        current_q1, current_q2 = self.critic(obs_t, act_t, hlc)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        if self._wandb:
            wandb.log({'train_critic/loss': critic_loss})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs: list, hlc: int):

        obs_t = to_tensor([o["affordances"] for o in obs], device=self.device)[0]

        # calculate sac loss
        action, log_prob = self.actor(obs_t, hlc)
        actor_q1, actor_q2 = self.critic(obs_t, action, hlc)
        actor_q = torch.min(actor_q1, actor_q2)
        sac_loss = (self.alpha.detach() * log_prob - actor_q).mean()

        if self._wandb:
            wandb.log({'train_actor/sac_loss': sac_loss.item()})
            wandb.log({'train_actor/target_entropy': self.target_entropy})
            wandb.log({'train_actor/entropy': -log_prob.mean().item()})

        # optimize the actor
        self.actor_optimizer.zero_grad()
        sac_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (- self.alpha * (log_prob + self.target_entropy).detach()).mean()
            if self._wandb:
                wandb.log({'train_alpha/loss': alpha_loss})
                wandb.log({'train_alpha/value': self.alpha})
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_actor_with_bc(self, loader: DataLoader, hlc: int):

        for i, (obs, act) in enumerate(loader):
            obs, act = to_tensor([o["encoding"] for o in obs], act, device=self.device)
            loss = self.actor.get_supervised_loss(obs, act, hlc)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            if self._wandb:
                wandb.log({'train_actor/bc_loss': loss})

    def update(self,
               replay_buffer: OnlineReplayBuffer,
               bc_loaders: Dict[str, DataLoader],
               step: int):

        self._step += 1
        for hlc in self.__hlc_to_train:
            online_samples = replay_buffer.sample(self.batch_size, hlc)
            obs, act, reward, next_obs, not_done = online_samples
            rewards = np.mean(reward)
            if self._wandb:
                wandb.log({f'train/batch_reward_{hlc}': rewards})
            self.update_critic(obs, act, reward, next_obs, not_done, hlc)

            if step % self.actor_bc_update_frequency == 0:
                self.update_actor_with_bc(bc_loaders[hlc], hlc)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, hlc)

            if step % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)

    def save(self, dirname: str, tag: str = 'best'):

        actor_filename = os.path.join(dirname, f"{tag}_actor.pth")
        critic_filename = os.path.join(dirname, f"{tag}_critic.pth")
        torch.save(self.actor.state_dict(), actor_filename)
        torch.save(self.critic_target.state_dict(), critic_filename)
        wandb.save(actor_filename)
        wandb.save(critic_filename)


if __name__ == "__main__":
    from models.carlaAffordancesDataset import AffordancesDataset, HLCAffordanceDataset
    from bc.train_bc import get_collate_fn
    from torch.utils.data import DataLoader
    import time

    data_path = "../data/"
    act_mode = "raw"
    number_of_test_updates = 100

    # pseudo-test --->
    obs_gen = lambda: dict(camera=None, affordances=np.random.rand(15), speed=np.random.rand(3) * 5,
                           hlc=int(np.random.rand() * 4))
    act_gen = lambda: np.random.rand(3 if act_mode == "raw" else 2)
    reward_gen = lambda: np.random.rand() * 3
    done_gen = lambda: True if np.random.rand() < 0.5 else False
    transition_tuple_gen = lambda: (obs_gen(), act_gen(), reward_gen(), obs_gen(), done_gen())

    # create fake online replay buffer
    buffer = OnlineReplayBuffer(8192)
    for _ in range(8192 * 3):
        buffer.add(*transition_tuple_gen())

    # create dummy data laoders (offline dataset)
    dataset = AffordancesDataset(data_path)
    custom_collate_fn = get_collate_fn(act_mode)
    bc_loaders = {hlc: DataLoader(HLCAffordanceDataset(dataset, hlc=hlc),
                                  batch_size=128,
                                  collate_fn=custom_collate_fn,
                                  shuffle=True) for hlc in [0, 1, 2, 3]}

    sac_agent = SACAgent(observation_dim=15,
                         action_dim=2 if act_mode == "pid" else 3,
                         action_range=(-1, 1))

    start = time.time()
    for i in range(number_of_test_updates):
        sac_agent.update(replay_buffer=buffer,
                         bc_loaders=bc_loaders,
                         step=i)
    total_time = time.time() - start
    print(f"{number_of_test_updates} updates in {total_time:.2f} seconds "
          f"(avg of {(total_time/number_of_test_updates):.2f} per step)")