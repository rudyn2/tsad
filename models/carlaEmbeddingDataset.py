import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PadSequence:
    def __call__(self, batch):
        # We assume that each element in "batch" is a tuple (sequence, action, next sequence element).
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.stack([s[2] for s in sorted_batch])
        actions = torch.stack([s[1] for s in sorted_batch])
        return sequences_padded, lengths, actions, labels


class CarlaEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path: str, json_path: str):
        self._path = embeddings_path
        self.json_path = json_path
        self._metadata = None
        self._timestamps = None
        self._timestamp2run = {}

        self.read_metadata()
        self.read_timestamps()

    def read_timestamps(self):

        with h5py.File(self._path, "r") as f:
            for run in f.keys():
                for timestamp in f[run].keys():
                    self._timestamp2run[timestamp] = run
        self._timestamps = list(self._timestamp2run.keys())

    def read_metadata(self):
        with open(self.json_path, "r") as f:
            self._metadata = json.load(f)

    def __getitem__(self, item):
        """
        select a window of size 4, whose its start index is the argument
        """
        t3 = self._timestamps[item]
        run_id = self._timestamp2run[t3]
        idx_t3 = list(self._metadata[run_id].keys()).index(t3)
        if idx_t3 > len(self._metadata[run_id]) - 6:
            idx_t3 = len(self._metadata[run_id]) - 6
        run_ts = list(self._metadata[run_id].keys())
        windows_ts = [t3] + run_ts[idx_t3+1: idx_t3+5]

        with h5py.File(self._path, "r") as f:
            embeddings = [np.array(f[run_id][t]) for t in windows_ts]
            control = self._metadata[run_id][windows_ts[-2]]['control']
            action = torch.tensor([control['steer'], control['throttle'], control['brake']])
            return torch.tensor(embeddings[:-1]), action, torch.tensor(embeddings[-1])

    def __len__(self):
        return sum([len(self._metadata[k]) // 4 for k in self._metadata.keys()])


class CarlaOnlineEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path: str, json_path: str):
        super(Dataset, self).__init__()
        self._source_dataset = CarlaEmbeddingDataset(embeddings_path, json_path)
        org_length = self._source_dataset
        self._sequences = [None] * len(org_length)
        self._actions = [None] * len(org_length)
        self._labels = [None] * len(org_length)
        self._load_all()

    def __getitem__(self, item):
        return self._sequences[item], self._actions[item], self._labels[item]

    def __len__(self):
        return len(self._sequences)

    def _load_all(self):
        for i in tqdm(range(len(self._source_dataset)), "Loading dataset"):
            s, a, s1 = self._source_dataset[i]
            self._sequences[i] = s
            self._actions[i] = a
            self._labels[i] = s1


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    d = CarlaEmbeddingDataset(embeddings_path='../embeddings.hdf5', json_path='../dataset/sample6.json')
    loader = DataLoader(d, batch_size=8, collate_fn=PadSequence(), drop_last=True)

    for batch in loader:
        print(batch[0].shape)
