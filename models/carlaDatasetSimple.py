from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import torch
from pathlib import Path


class CarlaDatasetSimple(Dataset):

    def __init__(self, path: str):
        self.path = path
        self.run_timestamp_mapping = {}
        self.timestamps = None
        self.metadata = None
        self.transform = None

        self.hdf5_path = self.read_timestamps()
        self.json_path = self.read_metadata()

    def read_timestamps(self):
        hdf5_files = list(Path(self.path).glob('**/*.hdf5'))
        if len(hdf5_files) < 1:
            raise RuntimeError("We couldn't find hdf5 files at provided path")

        # assume that there is just one hdf5 file at provided path
        hdf5_path = hdf5_files[0]
        with h5py.File(hdf5_path, "r") as f:
            for run in f.keys():
                for timestamp in f[run].keys():
                    self.run_timestamp_mapping[timestamp] = run
        self.timestamps = list(self.run_timestamp_mapping.keys())
        return hdf5_path

    def read_metadata(self):
        json_files = list(Path(self.path).glob('**/*.json'))
        if len(json_files) < 1:
            raise RuntimeError("We couldn't find json files at provided path")

        # assume that there is just one hdf5 file at provided path
        json_path = json_files[0]
        with open(json_path, "r") as f:
            self.metadata = json.load(f)
        return json_path

    def __getitem__(self, item):
        element = self.timestamps[item]
        run_id = self.run_timestamp_mapping[element]
        with h5py.File(self.path + ".hdf5", "r") as f:
            images = f[run_id][element]
            rgb = np.array(images['rgb'])
            depth = np.array(images['depth'])
            semantic = np.array(images['semantic'])

        data = self.metadata[run_id][element]
        x = np.concatenate((rgb, depth[:, :, np.newaxis]), axis=2)
        x = np.transpose(x, axes=[2, 0, 1])
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x).float()

        # get ground truth
        s = semantic[np.newaxis, :, :]
        s = torch.tensor(s, dtype=torch.int8)

        tl = torch.tensor([1, 0] if data['tl_state'] == 'Green' else [0, 1], dtype=torch.float16)
        v_aff = torch.tensor([data['lane_distance'], data['lane_orientation']]).float()

        return x, s, tl, v_aff

    def __len__(self):
        return len(self.run_timestamp_mapping)


if __name__ == '__main__':
    d = CarlaDatasetSimple('../dataset')

