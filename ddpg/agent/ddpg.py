import numpy as np
import torch
import torch.nn.functional as F
import wandb
import os

# https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py
class DDPG(object):
    def __init__(
        self, 
        actor,
        critic,
        actor_target,
        critic_target,

        state_dim, 
        action_dim: int = 2,
        action_range: tuple = (-1, 1),
        device: str = 'cuda',
        discount: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        actor_betas: tuple = (0.9, 0.999),
        actor_weight_decay: float = 4e-2,
        critic_betas: tuple = (0.9, 0.999),
        critic_weight_decay: float = 4e-2,
        batch_size: int = 1024,
        offline_proportion: float = 0.25,
        update_iteration: int = 200,
        ):

        self.action_range = action_range
        self.device = torch.device(device)
        self.gamma = discount
        self.tau = tau,
        self.batch_size = batch_size
        self.offline_proportion = offline_proportion
        self.update_iteration = update_iteration

        # Actor
        self.actor = actor
        self.actor_target = actor_target
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas,
                                                weight_decay=actor_weight_decay)
        self.actor.to(self.device)
        self.actor_target.to(self.device)

        # Critic
        self.critic = critic
        self.critic_target = critic_target
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas,
                                                 weight_decay=critic_weight_decay)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self._step = 0
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        for p in self.actor_target.parameters():
            p.requires_grad = False
        
        self.train()
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs):
        return self.actor(obs, obs['hlc'])

    def update(self, replay_buffer):
        self._step += 1
        hlc = 3

        for it in range(self.update_iteration):
            # Sample replay buffer
            online_samples, offline_samples = replay_buffer.sample(self.batch_size, self.offline_proportion)
            state, action, reward, next_state, not_done =online_samples
            done = torch.FloatTensor(1-not_done).to(self.device)
            offline_obs, offline_act, _, _, _ = offline_samples

            rewards = np.mean(reward[hlc])
            wandb.log({'train/batch_reward': rewards})

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            wandb.log({
                'Loss/critic_loss': critic_loss,
                'Loss/actor_loss': actor_loss,
            })

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self, dirname: str, tag: str = 'best'):

        actor_filename = os.path.join(dirname, f"{tag}_actor.pth")
        actor_target_filename = os.path.join(dirname, f"{tag}_actor_target.pth")
        critic_filename = os.path.join(dirname, f"{tag}_critic.pth")
        critic_target_filename = os.path.join(dirname, f"{tag}_critic_target.pth")
        torch.save(self.actor.state_dict(), actor_filename)
        torch.save(self.actor_target.state_dict(), actor_target_filename)
        torch.save(self.critic.state_dict(), critic_filename)
        torch.save(self.critic_target.state_dict(), critic_target_filename)
        wandb.save(actor_filename)
        wandb.save(critic_filename)