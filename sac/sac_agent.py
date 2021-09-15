from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import sac.utils as utils
from agents.agent import MultiTaskActor, MultiTaskCritic
from agents.squashed_gaussian import SquashedGaussianMLP, mlp
import wandb
import os
import torch.nn as nn
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from sac.replay_buffer import OnlineReplayBuffer


# https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac

class SACActor(MultiTaskActor, nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: tuple,
                 output_dim: int):
        super().__init__()

        self._actors = nn.ModuleDict({
            str(hlc): SquashedGaussianMLP(
                input_dim,
                output_dim,
                hidden_dim,
                nn.ReLU
            ) for hlc in range(4)
        })

    def act_batch(self, obs: list, task: int) -> Union[list, torch.Tensor]:
        pass

    def act_single(self, obs: dict, task: int) -> list:
        pass

    def forward(self, obs: list, hlc: int):
        """
        Returns actions and log-probability over those actions conditioned to the observations.
        For action selection please use act_batch or act_single.
        """
        encoding = torch.stack([torch.tensor(o['encoding'], device=self._device) for o in obs], dim=0).float()
        pi_distribution = self._actor[str(hlc)].get_distribution(encoding)
        pi_action = pi_distribution.rsample()
        pi_action = torch.tanh(pi_action)
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        return pi_action, logp_pi

    def supervised_update(self, obs: list, act: list, task: int) -> float:
        pass

    def train_mode(self):
        self._actors.train()

    def eval_mode(self):
        self._actors.eval()

    def save(self):
        pass

    def load(self):
        pass


class SACCritic(MultiTaskCritic, nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: tuple,
                 output_dim: int):
        super().__init__()

        self._critics1 = nn.ModuleDict({
            str(hlc): mlp([input_dim] + list(hidden_dim) + [output_dim], nn.ReLU) for hlc in range(4)
        })

        self._critics2 = nn.ModuleDict({
            str(hlc): mlp([input_dim] + list(hidden_dim) + [output_dim], nn.ReLU) for hlc in range(4)
        })

    def forward(self, obs, act, hlc) -> Tuple[torch.Tensor, torch.Tensor]:
        assert obs.size(0) == act.size(0)
        obs_action = torch.cat([obs, act], dim=1)
        q1 = self._critics1[hlc](obs_action)
        q2 = self._critics2[hlc](obs_action)
        return q1, q2

    def train_mode(self):
        self._critics1.train()
        self._critics2.train()

    def eval_mode(self):
        self._critics1.eval()
        self._critics2.eval()

    def save(self):
        pass

    def load(self):
        pass


class SACAgent(object):
    """SAC algorithm."""

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,  # used only for target entropy definition
                 action_range: tuple,
                 device: str = 'cuda',
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

        # store actor and critic
        self.critic = SACCritic(input_dim=observation_dim+action_dim, hidden_dim=(512, 512), output_dim=1)
        self.critic_target = SACCritic(input_dim=observation_dim+action_dim, hidden_dim=(512, 512), output_dim=1)
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
        self.__hlc_to_train = [0, 1, 2, 3]

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

        dist = self.actor(next_obs, hlc=hlc)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=False)
        target_q1, target_q2 = self.critic_target(next_obs[hlc], next_action, hlc=hlc)
        target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * log_prob
        target_q = reward[hlc] + (not_done[hlc] * self.discount * target_v).float()
        target_q = target_q.detach()
        current_q1, current_q2 = self.critic(obs[hlc], act[hlc].float(), hlc=hlc)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        wandb.log({'train_critic/loss': critic_loss})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs: list, hlc: int):

        action, log_prob = self.actor(obs, hlc=hlc)
        actor_q1, actor_q2 = self.critic.critic_batch(obs, action, hlc)
        actor_q = torch.min(actor_q1, actor_q2)
        sac_loss = (self.alpha.detach() * log_prob - actor_q).mean()
        log_prob = log_prob.mean()

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
            wandb.log({'train_alpha/loss': alpha_loss})
            wandb.log({'train_alpha/value': self.alpha})
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_actor_with_bc(self, loader: DataLoader, hlc: int):

        for i, (obs, act) in loader:
            loss = self.actor.supervised_update(obs, act, hlc)
            wandb.log({'train_actor/bc_loss': loss})

    def update(self,
               replay_buffer: OnlineReplayBuffer,
               bc_loaders: Dict[str, DataLoader],
               step: int):

        self._step += 1
        for hlc in self.__hlc_to_train:
            online_samples = replay_buffer.sample(self.batch_size, hlc)
            obs, act, reward, next_obs, not_done = online_samples
            rewards = np.mean(reward[hlc])
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
