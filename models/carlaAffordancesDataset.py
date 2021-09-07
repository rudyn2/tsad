import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


class AffordancesDataset(Dataset):
    def __init__(self, data_folder: str) -> None:
        super().__init__()
        self._data_folder = data_folder
        self._data_cache = {}
        self._timestamps_list = []

        p = Path(data_folder)
        assert(p.is_dir())

        self._metadata_files = sorted(p.glob('**/**/*.json'))
        
        for path in tqdm(self._metadata_files):
            with open(path) as f:
                metadata = json.load(f)
                self._push_data(metadata, path)
            del metadata
        

    def _get_episode_file(self, episode, metadata):
        date = metadata.name.split('.')[0]
        path = f"{metadata.parent}/{date}_{episode}.npz"
        p = Path(path)
        if p.is_file:
            return p
        print(f"Warning: data file not found {path}")

    def _push_data(self, metadata, metadata_path):
        for ep_key, episode in metadata.items():
            data_path = self._get_episode_file(ep_key, metadata_path)
            self._data_cache[ep_key] = episode
            episode_data = np.load(data_path)
            for t_key in episode.keys():
                affordances = episode_data[t_key]
                self._data_cache[ep_key][t_key]['affordances'] = affordances
                self._timestamps_list.append({
                    'episode': ep_key,
                    'timestamp': t_key,
                })

    def __getitem__(self, index: int):
        timestamp = self._timestamps_list[index]
        ep_key, t_key = timestamp['episode'], timestamp['timestamp']

        affordances = torch.tensor(self._data_cache[ep_key][t_key]['affordances']).float()

        steer = self._data_cache[ep_key][t_key]['control']['steer']
        brake = self._data_cache[ep_key][t_key]['control']['brake']
        throttle = self._data_cache[ep_key][t_key]['control']['throttle']
        control = torch.tensor([steer, throttle, brake]).float()
        
        command = self._data_cache[ep_key][t_key]['command']

        return affordances, control, command
    
    def __len__(self):
        return len(self._timestamps_list)


if __name__ == "__main__":
    path = '/home/johnny/Descargas/data'
    dataset = AffordancesDataset(path)
    dataset[10]