import torch
import wandb
import sys
from torch.utils.data import DataLoader

from sac.agent.actor import DiagGaussianActor
from models.carlaAffordancesDataset import AffordancesDataset, HLCAffordanceDataset

HLC_TO_NUMBER = {
    'RIGHT': 0,
    'LEFT': 1,
    'STRAIGHT': 2,
    'LANEFOLLOW': 3
}


def group_batches_by_hlc(samples: list) -> (dict, dict):
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
                 lr: float = 0.0001,
                 batch_size: int = 128,
                 epochs: int = 100,
                 eval_frequency: int = 20,
                 eval_episodes: int = 3,
                 use_wandb: bool = False
                 ):
        self._actor = actor
        self._dataset = dataset
        self._epochs = epochs
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=lr)
        self._wandb = use_wandb
        self._eval_frequency = eval_frequency
        self._eval_episode = eval_episodes

        self._batch_size = batch_size
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def eval(self):
        print("Not implemented yet.")

    def run(self):
        if self._wandb:
            wandb.init(project='tsad', entity='autonomous-driving')

        train_loaders = {hlc: DataLoader(HLCAffordanceDataset(self._dataset, hlc=hlc),
                                         batch_size=self._batch_size, collate_fn=group_batches_by_hlc,
                                         shuffle=True) for hlc in range(4)}
        for e in range(self._epochs):
            for hlc in range(4):
                hlc_loader = train_loaders[hlc]
                for i, (obs, act) in enumerate(hlc_loader):

                    dist_e = self._actor(obs, hlc=hlc)
                    act_e_hlc = torch.stack(act).float()
                    log_prob_e = dist_e.log_prob(torch.clamp(act_e_hlc, min=-1 + 1e-6, max=1.0 - 1e-6)).sum(-1,
                                                                                                            keepdim=True)
                    bc_loss = - log_prob_e.mean()
                    if self._wandb:
                        wandb.log({f'train_actor/bc_loss_{hlc}': bc_loss.item()})

                    self._actor_optimizer.zero_grad()
                    bc_loss.backward()
                    self._actor_optimizer.step()

                    sys.stdout.write("\r")
                    sys.stdout.write(f"Epoch={e}(hlc={hlc}) [{i}/{len(hlc_loader)}] bc_loss={bc_loss.item():.2f}")
                    sys.stdout.flush()

            if e % self._eval_frequency == 0:
                self.eval()


if __name__ == '__main__':
    actor = DiagGaussianActor(input_size=15, hidden_dim=64, action_dim=3, log_std_bounds=(-2, 5))
    dataset = AffordancesDataset('/Users/rudy/Documents/affordances/batch_1')
    trainer = BCTrainer(actor, dataset, use_wandb=True)
    trainer.run()
