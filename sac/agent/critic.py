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

    def forward(self, obs, action, hlc):
        q1 = self.Q1(obs, action, hlc)
        q2 = self.Q2(obs, action, hlc)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

        # assert len(self.Q1) == len(self.Q2)
        # for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
        #     assert type(m1) == type(m2)
        #     if type(m1) is nn.Linear:
        #         logger.log_param(f'train_critic/q1_fc{i}', m1, step)
        #         logger.log_param(f'train_critic/q2_fc{i}', m2, step)
