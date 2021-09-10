import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from collections import defaultdict


HLC_TO_NUMBER = {
        'RIGHT': 0,
        'LEFT': 1,
        'STRAIGHT': 2,
        'LANEFOLLOW': 3
    }


class AffordancesDataset(object):
    def __init__(self, data_folder: str) -> None:
        super().__init__()
        self._data_folder = data_folder
        self._data_cache = {}
        self.timestamps_lists = defaultdict(list)

        p = Path(data_folder)
        assert(p.is_dir())

        self._metadata_files = sorted(p.glob('**/**/*.json'))
        
        for path in tqdm(self._metadata_files, "Reading dataset..."):
            with open(path) as f:
                metadata = json.load(f)
                self._push_data(metadata, path)

    def _get_episode_file(self, episode, metadata):
        date = metadata.name.split('.')[0]
        path = f"{metadata.parent}/{date}_{episode}.npz"
        p = Path(path)
        if p.is_file:
            return p
        print(f"Warning: data file not found {path}")

    def _push_data(self, metadata, metadata_path):
        for ep_key, episode_metadata in metadata.items():
            data_path = self._get_episode_file(ep_key, metadata_path)
            self._data_cache[ep_key] = episode_metadata
            episode_data = np.load(data_path)
            for t_key in episode_metadata.keys():
                affordances = episode_data[t_key]
                hlc = HLC_TO_NUMBER[episode_metadata[t_key]['command']]
                self._data_cache[ep_key][t_key]['affordances'] = affordances
                self.timestamps_lists[hlc].append({
                    'episode': ep_key,
                    'timestamp': t_key,
                })

    def get_item(self, index: int, hlc: int):
        timestamp = self.timestamps_lists[hlc][index]
        ep_key, t_key = timestamp['episode'], timestamp['timestamp']

        affordances = self._data_cache[ep_key][t_key]['affordances']

        steer = self._data_cache[ep_key][t_key]['control']['steer']
        brake = self._data_cache[ep_key][t_key]['control']['brake']
        throttle = self._data_cache[ep_key][t_key]['control']['throttle']
        control = np.array([throttle, brake, steer])
        
        command = self._data_cache[ep_key][t_key]['command']

        return affordances, control, command
    
    def __len__(self):
        return sum([len(t_list) for t_list in self.timestamps_lists.values()])


class HLCAffordanceDataset(Dataset):

    def __init__(self,
                 affordance_dataset: AffordancesDataset,
                 hlc: int):
        self._dataset = affordance_dataset
        self._hlc = hlc

    def __getitem__(self, index: int):
        return self._dataset.get_item(index, self._hlc)

    def __len__(self):
        return len(self._dataset.timestamps_lists[self._hlc])


if __name__ == "__main__":
    path = '/Users/rudy/Documents/affordances/'
    dataset = AffordancesDataset(path)
    hlc_dataset = HLCAffordanceDataset(dataset, hlc=3)
    item = hlc_dataset[0]