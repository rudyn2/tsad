from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import torch
from pathlib import Path
import random
from torchvision.transforms import transforms


class CarlaDatasetSimple(Dataset):

    # moving obstacles (0),  traffic lights (1),  road markers(2),  road (3),  sidewalk (4) and background (5).
    CLASS_MAPPING = {
        0: 5,  # None
        1: 5,  # Buildings
        2: 5,  # Fences
        3: 5,  # Other
        4: 6,  # Pedestrians
        5: 5,  # Poles
        6: 2,  # RoadLines
        7: 3,  # Roads
        8: 4,  # Sidewalks
        9: 5,  # Vegetation
        10: 0,  # Vehicles
        11: 5,  # Walls
        12: 5,  # TrafficSigns
        13: 5,  # Sky
        14: 5,  # Ground
        15: 5,  # Bridge
        16: 5,  # RailTrack
        17: 5,  # GuardRail
        18: 1,  # Traffic Light
        19: 5,  # Static
        20: 5,  # Dynamic
        21: 5,  # Water
        22: 5  # Terrain
    }

    def __init__(self, path: str):
        self.path = path
        self.run_timestamp_mapping = {}
        self.timestamps = None
        self.metadata = None
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.depth_transform = None

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
        # self.timestamps = self.timestamps[:128]     # REMOVE THIS
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

    def get_metadata(self):
        return self.metadata

    def get_timestamps(self):
        return self.timestamps

    def get_random_full_episode(self):
        # select random episode
        run_id = random.choice(list(self.metadata.keys()))
        timestamps = self.metadata[run_id].keys()
        idxs = [self.timestamps.index(t) for t in timestamps]
        camera, segmentation, traffic_light, vehicle_affordances, pedestrian = [], [], [], [], []
        for idx in sorted(idxs):
            x, s, tl, v_aff, pds = self[idx]
            camera.append(x)
            segmentation.append(s)
            traffic_light.append(tl)
            vehicle_affordances.append(v_aff)
            pedestrian.append(pds)
        return camera, segmentation, traffic_light, vehicle_affordances, pedestrian

    def __getitem__(self, item):
        element = self.timestamps[item]
        run_id = self.run_timestamp_mapping[element]
        with h5py.File(self.hdf5_path, "r") as f:
            images = f[run_id][element]
            rgb = np.array(images['rgb'])
            depth = np.array(images['depth'])
            semantic = np.array(images['semantic'])

        data = self.metadata[run_id][element]
        # x = np.concatenate((rgb, depth[:, :, np.newaxis]), axis=2)
        # x = np.transpose(x, axes=[2, 0, 1])
        if self.transform:
            rgb_transformed = self.transform(rgb)
        else:
            rgb_transformed = torch.from_numpy(rgb).float()
        if self.depth_transform:
            depth_transformed = self.depth_transform(depth)
        else:
            depth_transformed = torch.tensor(depth).unsqueeze(0) / 1000

        # stack depth
        x = torch.cat([depth_transformed.float(), rgb_transformed]).float()

        # get ground truth
        s = semantic[np.newaxis, :, :]
        s = torch.tensor(s, dtype=torch.int8)
        s = self._map_classes(s)

        tl = torch.tensor([1, 0] if data['tl_state'] == 'Green' else [0, 1], dtype=torch.float16)
        v_aff = torch.tensor([data['lane_distance'], data['lane_orientation']]).float()
        sum_pds = (s == 6).sum()
        pds = torch.tensor([0, 1] if sum_pds > 100 else [1, 0]).float()

        return x, s, tl, v_aff, pds

    def _map_classes(self, semantic: torch.Tensor) -> torch.Tensor:
        mapped_semantic = torch.zeros_like(semantic)
        for k, v in self.CLASS_MAPPING.items():
            mapped_semantic[semantic == k] = v
        return mapped_semantic

    def __len__(self):
        return len(self.timestamps)


if __name__ == '__main__':
    d = CarlaDatasetSimple('../dataset')
    print(d.get_random_full_episode())

