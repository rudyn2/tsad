import json
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import defaultdict

#from utils.utils import normalize_action, normalize_speed
from gym_carla.envs.utils import normalize_action, normalize_speed

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
        self.timestamps_lists = defaultdict(list)       # dict of list of dict with episode-timestamp keys
        self.validation_episodes = list()               # list of episode keys

        self._total_validation_episodes = 2

        p = Path(data_folder)
        assert(p.is_dir())

        self._metadata_files = sorted(p.glob('**/**/*.json'))
        self._total_validation_episodes = len(self._metadata_files)

        print(f"Found {len(self._metadata_files)} metadata files")
        for path in tqdm(self._metadata_files, "Reading dataset..."):
            with open(path) as f:
                metadata = json.load(f)
                self._push_data(metadata, path)
        print(f"Total train timestamps: {sum([len(t_list) for t_list in self.timestamps_lists.values()])}")
        print(f"Total validation episodes: {len(self.validation_episodes)}")

    def _get_episode_file(self, episode, metadata):
        date = metadata.name.split('.')[0]
        path = f"{metadata.parent}/{date}_{episode}.npz"
        p = Path(path)
        if p.is_file:
            return p
        print(f"Warning: data file not found {path}")

    def _push_data(self, metadata, metadata_path):
        # randomly select one episode for validation
        # all these idxs will be stored as validation episodes
        val_episode_idx = random.choice(range(len(metadata)))

        for i, (ep_key, episode_metadata) in enumerate(metadata.items()):
            self._data_cache[ep_key] = episode_metadata
            if i == val_episode_idx:
                self.load_episode(ep_key, episode_metadata, metadata_path, save_idx=False)
                self.validation_episodes.append(ep_key)
                continue
            self.load_episode(ep_key, episode_metadata, metadata_path, save_idx=True)

    def load_episode(self, ep_key: str, episode_metadata: dict, metadata_path: str, save_idx: bool = True):
        data_path = self._get_episode_file(ep_key, metadata_path)
        episode_data = np.load(data_path)
        for t_key in episode_metadata.keys():
            affordances = episode_data[t_key]
            hlc = HLC_TO_NUMBER[episode_metadata[t_key]['command']]
            self._data_cache[ep_key][t_key]['affordances'] = affordances
            if save_idx:
                idx_element = dict(episode=ep_key, timestamp=t_key)
                self.timestamps_lists[hlc].append(idx_element)

    def get_episode(self, source: str, normalize_control: bool = True):
        """
        Get a tuple of lists with ordered data elements from an entire episode.
        """
        ep_key = None
        if source == "train":
            # find a train episode
            found = False
            while not found:
                ep_key = random.choice(list(self._data_cache.keys()))
                if ep_key not in self.validation_episodes:
                    break
            if not found:
                raise RuntimeError("Couldn't find a train episode")
        else:
            ep_key = random.choice(self.validation_episodes)

        timestamps = sorted(map(int, list(self._data_cache[ep_key].keys())))    # ensure time order
        print(f"Selected episode: {ep_key} with {len(timestamps)} frames")
        ep_aff, ep_control, ep_speed, ep_command = [], [], [], []
        for timestamp in timestamps:
            affordances, control, speed, command = self.unpack_data(ep_key, str(timestamp))

            if normalize_control:
                control = normalize_action(control)
                speed = normalize_speed(speed)

            ep_aff.append(affordances)
            ep_control.append(control)
            ep_speed.append(speed)
            ep_command.append(command)

        return ep_aff, ep_control, ep_speed, ep_command

    def get_item(self, index: int, hlc: int, use_next_speed: bool = False, normalize_control: bool = False):
        timestamp = self.timestamps_lists[hlc][index]
        ep_key, t_key = timestamp['episode'], timestamp['timestamp']

        affordances, control, speed, command = self.unpack_data(ep_key, t_key)
        if normalize_control:
            control = normalize_action(control)
            speed = normalize_speed(speed)
        
        if use_next_speed:
            next_speed = self.get_next_speed(index, ep_key, hlc)
            if normalize_control:
                next_speed = normalize_speed(next_speed)
            return affordances, control, speed, command, next_speed
        return affordances, control, speed, command
    
    def unpack_data(self, episode, timestamp):
        affordances = self._data_cache[episode][timestamp]['affordances']
        steer = self._data_cache[episode][timestamp]['control']['steer']
        brake = self._data_cache[episode][timestamp]['control']['brake']
        throttle = self._data_cache[episode][timestamp]['control']['throttle']
        control = np.array([throttle, brake, steer])
        speed = self._data_cache[episode][timestamp]['speed']
        command = self._data_cache[episode][timestamp]['command']
        return affordances, control, speed, command
    
    def get_next_speed(self, index, episode, hlc):
        next_index = index + 1
        if next_index < len(self.timestamps_lists[hlc]):
            timestamp = self.timestamps_lists[hlc][next_index]
            ep_key, t_key = timestamp['episode'], timestamp['timestamp']
            # next timestamp in database is the next timestamp in episode, return next speed
            if ep_key == episode:
                return self._data_cache[ep_key][t_key]['speed']
        # actual timestamp is the last in the episode or the last in database, return actual speed
        else:
            timestamp = self.timestamps_lists[hlc][index]
            ep_key, t_key = timestamp['episode'], timestamp['timestamp']
            return self._data_cache[ep_key][t_key]['speed']

    def __len__(self):
        return sum([len(t_list) for t_list in self.timestamps_lists.values()])


class HLCAffordanceDataset(Dataset):

    def __init__(self,
                 affordance_dataset: AffordancesDataset,
                 hlc: int,
                 use_next_speed: bool = False,
                 normalize_control: bool = False
                 ):
        self._dataset = affordance_dataset
        self._hlc = hlc
        self._use_next_speed = use_next_speed
        self._normalize_control = normalize_control

    def __getitem__(self, index: int):
        return self._dataset.get_item(index, self._hlc, self._use_next_speed, self._normalize_control)

    def __len__(self):
        return len(self._dataset.timestamps_lists[self._hlc])


def plot_steer_histogram(dataset: HLCAffordanceDataset):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    for hlc, ax in zip(range(4), axs.reshape(-1)):
        hlc_dataset = HLCAffordanceDataset(dataset, hlc)
        steer_values = []
        for index in range(len(hlc_dataset)):
            steer_values.append(hlc_dataset[index][1][2])
        ax.hist(steer_values, bins=40, range=(-1, 1), density=True)
        ax.set_title(f"Histogram of steering values (HLC={hlc})")
        ax.set_xlabel("Steering")
    plt.show()


if __name__ == "__main__":
    path = '/home/rudy/Documents/tsad/data/small'
    dataset = AffordancesDataset(path)
    ep_aff, ep_ctrl, ep_speed, ep_cmd = dataset.get_episode(source="val", normalize_control=True)
    plot_steer_histogram(dataset)