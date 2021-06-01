from torch import nn

import sac.utils as utils
from models.ActorCritic import Critic


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, action_dim, hidden_dim):
        super().__init__()

        self.Q1 = Critic(hidden_dim=hidden_dim, action_dim=action_dim)
        self.Q2 = Critic(hidden_dim=hidden_dim, action_dim=action_dim)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        # assert len(self.Q1) == len(self.Q2)
        # for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
        #     assert type(m1) == type(m2)
        #     if type(m1) is nn.Linear:
        #         logger.log_param(f'train_critic/q1_fc{i}', m1, step)
        #         logger.log_param(f'train_critic/q2_fc{i}', m2, step)
