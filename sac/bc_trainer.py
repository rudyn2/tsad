from termcolor import colored
import warnings
import wandb
import torch
from gym_carla.envs.carla_env import CarlaEnv
from sac.agent.actor import DiagGaussianActor
from sac.replay_buffer import OfflineBuffer
import argparse


class BCTrainer(object):

    def __init__(self,
                 actor: DiagGaussianActor,
                 offline_buffer: OfflineBuffer,
                 lr: float = 0.0001,
                 batch_size: int = 64,
                 steps: int = 1000,
                 use_wandb: bool = False
                 ):
        self._actor = actor
        self._offline_buffer = offline_buffer
        self._steps = steps
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=lr)
        self._wandb = use_wandb

        self._batch_size = batch_size
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def update_actor_with_bc(self, obs, act):
        hlc = 3
        dist_e = self._actor(obs[hlc], hlc=hlc)
        act_e_hlc = self.act_parser_invert(torch.as_tensor(act[hlc], device=torch.device(self._device)).float())
        log_prob_e = dist_e.log_prob(torch.clamp(act_e_hlc, min=-1 + 1e-6, max=1.0 - 1e-6)).sum(-1, keepdim=True)
        bc_loss = - log_prob_e.mean()
        if self._wandb:
            wandb.log({'train_actor/bc_loss': bc_loss.item()})
        self._actor_optimizer.zero_grad()
        bc_loss.backward()

    def act_parser(self, two_dim_action: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(two_dim_action.shape[0], 3)
        output[:, 0] = two_dim_action[:, 0]  # copy throttle-brake
        output[:, 1] = -two_dim_action[:, 0]  # copy throttle-brake
        output[:, 2] = two_dim_action[:, 1]  # copy steer

        # clamp the actions
        output = torch.max(torch.min(output, torch.tensor([[1, 1., 1]])), torch.tensor([[0, 0, -1.]]))
        output = output.to(self._device)
        return output.float()

    def act_parser_invert(self, three_dim_action: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(three_dim_action.shape[0], 2)
        output[:, 0] = three_dim_action[:, 0] - three_dim_action[:, 1]      # throttle - brake
        output[:, 1] = three_dim_action[:, 2]   # steer
        output = torch.clamp(output, min=-1, max=1)
        output = output.to(self._device)
        return output.detach().float()

    def run(self):
        for i in range(1, self._steps + 1):
            offline_obs, offline_act = self._offline_buffer.sample(self._batch_size)
            self.update_actor_with_bc(offline_obs, offline_act)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BC Trainer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default=None)

    parser.add_argument('--actor-hidden-dim', type=int, default=128, help='Size of hidden layer in the '
                                                                          'actor model.')
    parser.add_argument('--actor-weights', type=str, default=None, help='Path to actor weights')
    parser.add_argument('--batch-size', default=1024, type=int, help='Batch size.')
    parser.add_argument('--debug', action='store_true', help='Whether or not visualize actor input')
    parser.add_argument('--num-eval-episodes', default=3, type=int)
    parser.add_argument('--wandb', action='store_true')

    # carla parameters
    carla_config = parser.add_argument_group('CARLA config')
    carla_config.add_argument('--host', default='172.18.0.1', type=str, help='IP address of CARLA host.')
    carla_config.add_argument('--port', default=2008, type=int, help='Port number of CARLA host.')
    carla_config.add_argument('--vehicles', default=100, type=int, help='Number of vehicles in the simulation.')
    carla_config.add_argument('--walkers', default=50, type=int, help='Number of walkers in the simulation.')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    control_action_dim = 2
    offline_dataset_path = args.data

    actor = DiagGaussianActor(input_size=768*4*4,
                              action_dim=control_action_dim,
                              hidden_dim=args.actor_hidden_dim,
                              log_std_bounds=(-2, 5))
    if args.actor_weights:
        actor.load_state_dict(torch.load(args.actor_weights))
    actor.to(device)
    actor.train()

    print(colored("Training", "white"))
    # wandb.init(project='tsad', entity='autonomous-driving')
    trainer = BCTrainer(actor=actor,
                        offline_buffer=OfflineBuffer(offline_dataset_path + ".hdf5", offline_dataset_path + ".json"),
                        use_wandb=args.wandb)
    trainer.run()
