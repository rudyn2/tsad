import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CarlaEncodingDataset(Dataset):
    """
    It delivers a frame, the action took at that frame and the current speed.
    """

    def __init__(self, hdf5_path: str, json_path: str):
        self._path = hdf5_path
        self.json_path = json_path
        self._metadata = None
        self._timestamps = None
        self._timestamp2run = {}

        self.read_metadata()
        self.read_timestamps()

        self.to_tensor = T.ToTensor()

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
        ts = self._timestamps[item]
        run_id = self._timestamp2run[ts]

        with h5py.File(self._path, "r") as f:
            rgb = np.array(f[run_id][ts]['rgb'])
            depth = np.array(f[run_id][ts]['depth'])

            # process and stack
            rgb_transformed = self.to_tensor(rgb)
            depth_transformed = torch.tensor(depth).unsqueeze(0) / 1000
            input_ = torch.cat([depth_transformed, rgb_transformed]).float()
            input_ = input_[:, 64:, :]

            metadata = self._metadata[run_id][ts]
            action = torch.tensor([metadata['control']['steer'], metadata['control']['throttle'], metadata['control']['brake']])
            speed = torch.tensor([metadata['speed_x'], metadata['speed_y'], metadata['speed_y']])

            return input_, action, speed, (run_id, ts)

    def __len__(self):
        return len(self._timestamps)


if __name__ == '__main__':
    d = CarlaEncodingDataset(hdf5_path=r'C:\Users\C0101\Documents\tsad\dataset\data\batch_local_0.hdf5',
                             json_path=r'C:\Users\C0101\Documents\tsad\dataset\data\batch_local_0.json')
    e = d[0]
