import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sac.utils as utils
from . import Agent
from sac.agent.actor import DiagGaussianActor
from sac.agent.critic import DoubleQCritic
from sac.eps_scheduler import Epsilon
import wandb
import os


class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(self,
                 actor: DiagGaussianActor,
                 critic: DoubleQCritic,
                 target_critic: DoubleQCritic,
                 action_dim: int = 2,  # used only for target entropy definition
                 action_range: tuple = (-1, 1),
                 device: str = 'cuda',
                 discount: float = 0.99,
                 init_temperature: float = 0.3,
                 critic_tau: float = 0.005,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 alpha_lr: float = 1e-4,
                 actor_betas: tuple = (0.9, 0.999),
                 actor_weight_decay: float = 4e-2,
                 critic_betas: tuple = (0.9, 0.999),
                 critic_weight_decay: float = 4e-2,
                 alpha_betas: tuple = (0.9, 0.999),
                 actor_update_frequency: int = 1,
                 critic_target_update_frequency: int = 2,
                 batch_size: int = 1024,
                 actor_bc_update_frequency: int = 4,
                 offline_proportion: float = 0.25,
                 learnable_temperature: bool = True):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.actor_bc_update_frequency = actor_bc_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.offline_proportion = offline_proportion

        # store actor and critic
        self.critic = critic
        self.critic_target = target_critic
        # noinspection PyTypeChecker
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = actor

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
                                                betas=actor_betas,
                                                weight_decay=actor_weight_decay)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas,
                                                 weight_decay=critic_weight_decay)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()
        self._eps = Epsilon(100000, method="linear", factor=5)
        self._step = 0

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        # return torch.tensor(0.3 * self._eps(self._step), requires_grad=False)
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        dist = self.actor(obs, obs['hlc'])
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return utils.to_np(action[0])

    def act_parser(self, two_dim_action: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(two_dim_action.shape[0], 3)
        output[:, 0] = two_dim_action[:, 0]  # copy throttle-brake
        output[:, 1] = -two_dim_action[:, 0]  # copy throttle-brake
        output[:, 2] = two_dim_action[:, 1]  # copy steer

        # clamp the actions
        output = torch.max(torch.min(output, torch.tensor([[1, 1., 1]])), torch.tensor([[0, 0, -1.]]))
        output = output.to(self.device)
        return output.float()

    def act_parser_invert(self, three_dim_action: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(three_dim_action.shape[0], 2)
        output[:, 0] = three_dim_action[:, 0] - three_dim_action[:, 1]      # throttle - brake
        output[:, 1] = three_dim_action[:, 2]   # steer
        output = torch.clamp(output, min=-1, max=1)
        output = output.to(self.device)
        return output.detach().float()

    def _to_tensor(self, arr):
        return torch.as_tensor(arr, device=self.device).float()

    def update_critic(self, obs, act, reward, next_obs, not_done):

        hlc = 3
        dist = self.actor(next_obs[hlc], hlc=hlc)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=False)
        next_action = self.act_parser(next_action)
        target_Q1, target_Q2 = self.critic_target(next_obs[hlc], next_action, hlc=hlc)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = self._to_tensor(reward[hlc]) + (self._to_tensor(not_done[hlc]) * self.discount * target_V).float()
        target_Q = target_Q.detach()
        current_Q1, current_Q2 = self.critic(obs[hlc], self._to_tensor(act[hlc]).float(), hlc=hlc)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        wandb.log({'train_critic/loss': critic_loss})

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs):
        # HARDCODED: ONLY LANE FOLLOW TASK
        hlc = 3

        # with torch.cuda.amp.autocast(): -> mixed precision training
        # on-policy actor loss
        dist = self.actor(obs[hlc], hlc=hlc)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs[hlc], self.act_parser(action), hlc=hlc)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        sac_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        log_prob = log_prob.mean()

        wandb.log({'train_actor/sac_loss': sac_loss.item()})
        wandb.log({'train_actor/target_entropy': self.target_entropy})
        wandb.log({'train_actor/entropy': -log_prob.mean().item()})

        # optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        sac_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (- self.alpha * (log_prob + self.target_entropy).detach()).mean()
            wandb.log({'train_alpha/loss': alpha_loss})
            wandb.log({'train_alpha/value': self.alpha})
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_actor_with_bc(self, obs, act):
        hlc = 3
        dist_e = self.actor(obs[hlc], hlc=hlc)
        act_e_hlc = self.act_parser_invert(self._to_tensor(act[hlc]))
        log_prob_e = dist_e.log_prob(torch.clamp(act_e_hlc, min=-1 + 1e-6, max=1.0 - 1e-6)).sum(-1, keepdim=True)
        bc_loss = - log_prob_e.mean()
        wandb.log({'train_actor/bc_loss': bc_loss.item()})

        self.actor_optimizer.zero_grad(set_to_none=True)
        bc_loss.backward()
        self.actor_optimizer.step()

    def update(self, replay_buffer, step):
        self._step += 1
        hlc = 3
        online_samples, offline_samples = replay_buffer.sample(self.batch_size, self.offline_proportion)

        obs, act, reward, next_obs, not_done = online_samples
        offline_obs, offline_act, _, _, _ = offline_samples

        rewards = np.mean(reward[hlc])
        wandb.log({'train/batch_reward': rewards})

        self.update_critic(obs, act, reward, next_obs, not_done)

        # if step % self.actor_bc_update_frequency == 0:
        #     self.update_actor_with_bc(offline_obs, offline_act)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs)

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
