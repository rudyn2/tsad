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
        labels = torch.stack([s[3] for s in sorted_batch])
        speeds = torch.stack([s[2] for s in sorted_batch])
        actions = torch.stack([s[1] for s in sorted_batch])
        return sequences_padded, lengths, actions, speeds, labels


class CarlaEmbeddingDataset(Dataset):
    """
    It delivers sequences of length 4 as the
    """
    def __init__(self, embeddings_path: str, json_path: str, provide_ts: bool = False, sequence: bool = False):
        self._path = embeddings_path
        self.json_path = json_path
        self._metadata = None
        self._timestamps = None
        self._provide_ts = provide_ts
        self._timestamp2run = {}
        self._sequence = sequence

        self.read_metadata()
        self.read_timestamps()

    def read_timestamps(self):

        with h5py.File(self._path, "r") as f:
            for run in f.keys():
                for timestamp in f[run].keys():
                    self._timestamp2run[timestamp] = run
        self._timestamps = list(self._timestamp2run.keys())

        # just some episodes
        # self._timestamps = self._timestamps[:1000]
        # filtered_metadata = {}
        # for t in self._timestamps:
        #     run_id = self._timestamp2run[t]
        #     if run_id not in filtered_metadata.keys():
        #         filtered_metadata[run_id] = self._metadata[run_id]
        # self._metadata = filtered_metadata
        # print("")

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
            metadata = self._metadata[run_id][windows_ts[-2]]
            action = torch.tensor([control['steer'], control['throttle'], control['brake']])
            speed = torch.tensor([metadata['speed_x'], metadata['speed_y'], metadata['speed_y']])
            if self._provide_ts:
                return torch.tensor(embeddings[:-1]), action, speed, torch.tensor(embeddings[-1]), (run_id, windows_ts[-2])
            
            if self._sequence:
                return torch.tensor(embeddings[:-1]), action, speed, torch.tensor(embeddings[1:])
            else:
                return torch.tensor(embeddings[:-1]), action, speed, torch.tensor(embeddings[-1])

    def __len__(self):
        return sum([len(self._metadata[k]) // 4 for k in self._metadata.keys()])


class CarlaOnlineEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path: str, json_path: str, provide_ts: bool = False, sequence: bool = False):
        super(Dataset, self).__init__()
        self._source_dataset = CarlaEmbeddingDataset(embeddings_path, json_path, provide_ts, sequence=sequence)
        org_length = self._source_dataset
        self._provide_ts = provide_ts
        self._sequences = [None] * len(org_length)
        self._actions = [None] * len(org_length)
        self._speeds = [None] * len(org_length)
        self._labels = [None] * len(org_length)
        if self._provide_ts:
            self._ts = [None] * len(org_length)
        self._load_all()

    def __getitem__(self, item):
        if self._provide_ts:
            return self._sequences[item], self._actions[item], self._speeds[item], self._labels[item], self._ts[item]
        return self._sequences[item], self._actions[item], self._speeds[item], self._labels[item]

    def __len__(self):
        return len(self._sequences)

    def _load_all(self):
        for i in tqdm(range(len(self._source_dataset)), "Loading dataset"):
            if self._provide_ts:
                s, a, sps, s1, ts = self._source_dataset[i]
                self._ts[i] = ts
            else:
                s, a, sps, s1 = self._source_dataset[i]

            self._sequences[i] = s
            self._actions[i] = a
            self._speeds[i] = sps
            self._labels[i] = s1


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    d = CarlaOnlineEmbeddingDataset(embeddings_path='/home/johnny/Escritorio/batch_3.hdf5',
                                    json_path='/home/johnny/Escritorio/batch_3.json',
                                    sequence=True)
    loader = DataLoader(d, batch_size=8, collate_fn=PadSequence(), drop_last=True)

    for batch in loader:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)
        break
