import os
import wandb
import numpy as np

import torch
import torch.nn.functional as F


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/MoritzTaylor/ddpg-pytorch
class DDPG(object):

    def __init__(
        self, 
        actor,
        critic,
        target_actor,
        target_critic,
        action_range: tuple = (-1, 1),
        device: str = 'cuda',
        discount: float = 0.99,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        tau: float = 0.005, 
        actor_betas: tuple = (0.9, 0.999),
        actor_weight_decay: float = 4e-2,
        critic_betas: tuple = (0.9, 0.999),
        critic_weight_decay: float = 4e-2,
        batch_size: int = 1024,
        offline_proportion: float = 0.25,
        ):
        """DDPG constructor

        Args:
            actor ([type]): [description]
            critic ([type]): [description]
            target_actor ([type]): [description]
            target_critic ([type]): [description]
            action_dim (int, optional): [description]. Defaults to 2.
            device (str, optional): [description]. Defaults to 'cuda'.
            discount (float, optional): [description]. Defaults to 0.99.
            actor_lr (float, optional): [description]. Defaults to 1e-4.
            critic_lr (float, optional): [description]. Defaults to 1e-4.
            tau (float, optional): [description]. Defaults to 0.005.
            actor_betas (tuple, optional): [description]. Defaults to (0.9, 0.999).
            actor_weight_decay (float, optional): [description]. Defaults to 4e-2.
            critic_betas (tuple, optional): [description]. Defaults to (0.9, 0.999).
            critic_weight_decay (float, optional): [description]. Defaults to 4e-2.
            batch_size (int, optional): [description]. Defaults to 1024.
            offline_proportion (float, optional): [description]. Defaults to 0.25.
        """        
        self.gamma = discount
        self.tau = tau
        self.action_space = action_range
        self.batch_size = batch_size
        self.offline_proportion = offline_proportion
        self.device = device

        # Define the actor
        self.actor = actor.to(device)
        self.actor_target = target_actor.to(device)

        # Define the critic
        self.critic = critic.to(device)
        self.critic_target = target_critic.to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas,
                                                weight_decay=actor_weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas,
                                                 weight_decay=critic_weight_decay)

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self._step = 0
    
    def reset(self):
        pass

    def train(self, training=True):
        """[summary]

        Args:
            training (bool, optional): [description]. Defaults to True.
        """        
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, state, action_noise=None):
        """Evaluates the action to perform in a given state

        Args:
            state ([dict]): State to perform the action on in the env. 
                            Used to evaluate the action.
                            {"encoding": array([15]), "speed": array([3]), "hlc": int}
            action_noise ([type], optional): If not None, the noise to apply on the evaluated action. Defaults to None.
        Returns:
            [type]: [description]
        """        
        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(state, state['hlc'])
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(self.device)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space[0], self.action_space[0])

        return mu

    def update(self, replay_buffer):
        """Updates the parameters/networks of the agent according to the given batch.
            This means we ...
                1. Compute the targets
                2. Update the Q-function/critic by one step of gradient descent
                3. Update the policy/actor by one step of gradient ascent
                4. Update the target networks through a soft update

        Args:
            replay_buffer ([type]): [description]

        Returns:
            [type]: [description]
        """        
        self._step += 1
        hlc = 3
        # Sample from replay buffer
        online_samples, offline_samples = replay_buffer.sample(self.batch_size, self.offline_proportion)

        # Get tensors from the batch
        state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = online_samples
        offline_obs, offline_act, _, _, _ = offline_samples

        # Retrieve only hlc=3 lane follow
        state_batch = state_batch[hlc]
        action_batch = self._to_tensor(action_batch[hlc]).float()
        reward_batch = self._to_tensor(reward_batch[hlc])
        next_state_batch = next_state_batch[hlc]
        not_done_batch = self._to_tensor(not_done_batch[hlc])
        done_batch = 1 - not_done_batch

        rewards = torch.mean(reward_batch)
        wandb.log({'train/batch_reward': rewards.cpu().data.numpy()})

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch, hlc)
        next_state_action_values = self.critic_target(next_state_batch, self.act_parser(next_action_batch.detach()), hlc)

        # Compute the target
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # TODO: Clipping the expected values here?
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch, hlc)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.act_parser(self.actor(state_batch, hlc)), hlc)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save(self, dirname: str, tag: str = 'best'):
        """[summary]

        Args:
            dirname (str): [description]
            tag (str, optional): [description]. Defaults to 'best'.
        """        
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
    
    def act_parser(self, two_dim_action: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            two_dim_action (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """        
        output = torch.zeros(two_dim_action.shape[0], 3)
        output[:, 0] = two_dim_action[:, 0]  # copy throttle-brake
        output[:, 1] = -two_dim_action[:, 0]  # copy throttle-brake
        output[:, 2] = two_dim_action[:, 1]  # copy steer

        # clamp the actions
        output = torch.max(torch.min(output, torch.tensor([[1, 1., 1]])), torch.tensor([[0, 0, -1.]]))
        output = output.to(self.device)
        return output.float()

    def act_parser_invert(self, three_dim_action: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            three_dim_action (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """        
        output = torch.zeros(three_dim_action.shape[0], 2)
        output[:, 0] = three_dim_action[:, 0] - three_dim_action[:, 1]      # throttle - brake
        output[:, 1] = three_dim_action[:, 2]   # steer
        output = torch.clamp(output, min=-1, max=1)
        output = output.to(self.device)
        return output.detach().float()
    
    def _to_tensor(self, arr):
        """[summary]

        Args:
            arr ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return torch.as_tensor(arr, device=self.device).float()