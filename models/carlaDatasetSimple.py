from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import torch


class CarlaDatasetSimple(Dataset):

    def __init__(self, path: str):
        self.path = path
        self.run_timestamp_mapping = {}
        self.timestamps = None
        self.metadata = None
        self.transform = None
        self.read_timestamps()
        self.read_metadata()

    def read_timestamps(self):
        with h5py.File(self.path + ".hdf5", "r") as f:
            for run in f.keys():
                for timestamp in f[run].keys():
                    self.run_timestamp_mapping[timestamp] = run
        self.timestamps = list(self.run_timestamp_mapping.keys())

    def read_metadata(self):
        with open(self.path + ".json", "r") as f:
            self.metadata = json.load(f)

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
        s = semantic[np.newaxis, :, :] - 1
        s = torch.tensor(s, dtype=torch.int8)

        tl = torch.tensor([1, 0] if data['tl_state'] == 'Green' else [0, 1], dtype=torch.float16)
        v_aff = torch.tensor([data['lane_distance'], data['lane_orientation']]).float()

        return x, s, tl, v_aff

    def __len__(self):
        return len(self.run_timestamp_mapping)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    train_dataset = CarlaDatasetSimple("/home/rudy/Documents/tsad/dataset/sample6")
    train_loader = DataLoader(train_dataset, batch_size=32)

    start = time.time()
    for img, semantic_map, tl_status, vehicle_aff in train_loader:
        img, semantic_map, tl_status, vehicle_aff = img.to('cuda'), semantic_map.to('cuda'), tl_status.to(
            'cuda'), vehicle_aff.to('cuda')
        load_batch_time = time.time() - start
        start = time.time()
        print(f"Load time: {load_batch_time}")
