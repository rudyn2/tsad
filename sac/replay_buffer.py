import json

import h5py
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict
from collections import defaultdict


class ReplayMemoryFast:

    # first we define init method and initialize buffer size
    def __init__(self, memory_size: int):

        # max number of samples to store
        self.memory_size: int = memory_size

        self.experience: list = [None] * self.memory_size
        self.current_index: int = 0
        self.size = 0

    def add_(self, observation, action):
        # store the experience as a tuple (current state, action, reward, next state, is it a terminal state)
        self.experience[int(self.current_index)] = (observation, action)
        self.current_index += 1

        self.size = min(self.size + 1, self.memory_size)

        # if the index is greater than  memory then we flush the index by subtrating it with memory size
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    # next we define the function called store for storing the experiences
    def add(self, observation, action, reward, newobservation, not_done):

        # store the experience as a tuple (current state, action, reward, next state, is it a terminal state)
        self.experience[int(self.current_index)] = (observation, action, reward, newobservation, not_done)
        self.current_index += 1

        self.size = min(self.size + 1, self.memory_size)

        # if the index is greater than  memory then we flush the index by subtrating it with memory size
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    # we define a function called sample for sampling a minibatch of experience
    def sample(self, batch_size: int) -> list:
        """
        Sample from buffer.
        :param batch_size: number of samples to collect.
        :type batch_size: int
        :return: a list of tuples (observation, action, reward, new_observation, not_done)
        :rtype: list
        """
        if self.size < batch_size:
            return []

        # first we randomly sample some indices
        samples_index = np.floor(np.random.random((batch_size,)) * self.size)

        # select the experience from the sampled index
        samples = [self.experience[int(i)] for i in samples_index]

        return list(zip(*samples))

    def unpacked_sample(self, batch_size: int) -> list:
        """
        Sample from buffer
        :param batch_size: number of samples to collect.
        :type batch_size: int
        :return: a list of tuples (observation, action, reward, new_observation, not_done)
        :rtype: list
        """
        if self.size < batch_size:
            return []

        # first we randomly sample some indices
        samples_index = np.floor(np.random.random((batch_size,)) * self.size)

        # select the experience from the sampled index
        samples = [self.experience[int(i)] for i in samples_index]
        return samples

    def __len__(self):
        return self.size


class OnlineReplayBuffer(object):
    HLC_TO_NUMBER = {
        'RIGHT': 0,
        'LEFT': 1,
        'STRAIGHT': 2,
        'LANEFOLLOW': 3
    }

    def __init__(self,  online_memory_size: int):

        # create buffer per each high level command
        self._online_buffers = {
            0: ReplayMemoryFast(online_memory_size),    # RIGHT
            1: ReplayMemoryFast(online_memory_size),    # LEFT
            2: ReplayMemoryFast(online_memory_size),    # STRAIGHT
            3: ReplayMemoryFast(online_memory_size)     # LANE FOLLOW
        }

    def sample(self, batch_size: int, hlc: int):
        return self._online_buffers[hlc].sample(batch_size)

    def add(self, obs, action, reward, next_obs, done):
        """
        Add a transition to the online buffer.
        """
        self._online_buffers[int(obs['hlc'])].add(obs, action, reward, next_obs, done)

    def log_status(self):
        s = "\nOnline buffers:\n"
        for k in self._online_buffers.keys():
            s += f"{k}: {len(self._online_buffers[k])} samples\n"
        return s


class OfflineBuffer:
    HLC_TO_NUMBER = {
        'RIGHT': 0,
        'LEFT': 1,
        'STRAIGHT': 2,
        'LANEFOLLOW': 3
    }

    def __init__(self,
                 offline_buffer_hdf5: str = None,
                 offline_buffer_json: str = None):
        self._offline_buffer_hdf5_path = offline_buffer_hdf5
        self._offline_buffer_json_path = offline_buffer_json
        self._offline_buffers = self._load()

    def _build_offline_buffers(self, metadata):
        buffers_length = defaultdict(int)
        with h5py.File(self._offline_buffer_hdf5_path, "r") as f:
            for run_id in f.keys():
                for timestep in f[run_id].keys():
                    buffers_length[self.HLC_TO_NUMBER[metadata[run_id][timestep]['command']]] += 1

        buffers = {k: ReplayMemoryFast(v) for k, v in buffers_length.items()}
        return buffers

    def _load(self) -> Dict[int, ReplayMemoryFast]:
        with open(self._offline_buffer_json_path, "r") as f:
            metadata = json.load(f)

        offline_buffers = self._build_offline_buffers(metadata)
        with h5py.File(self._offline_buffer_hdf5_path, "r") as f:
            for run_id in tqdm(list(f.keys()), "Loading dataset"):
                episode_metadata = metadata[run_id]
                steps = list(f[run_id].keys())
                steps = sorted(steps)  # to be sure that they are in order
                for idx in range(len(steps) - 1):
                    step = steps[idx]
                    transition = self._get_transition(f[run_id], episode_metadata, step)
                    # add to some buffer depending on the HLC command of the current observation
                    offline_buffers[transition[0]['hlc']].add_(*transition)
        return offline_buffers

    def _get_transition(self, h5py_group: h5py.File, metadata_json: dict, step: str) \
            -> Tuple[Dict[str, np.ndarray], np.ndarray]:

        encoding = np.array(h5py_group[step])
        step_metadata = metadata_json[step]
        step_speed = np.array([float(step_metadata['speed_x']), float(step_metadata['speed_y']),
                               float(step_metadata['speed_z'])])
        step_command = int(self.HLC_TO_NUMBER[step_metadata['command']])
        observation = dict(encoding=encoding, speed=step_speed, hlc=step_command)
        action = np.array([float(step_metadata['control']['throttle']),
                           float(step_metadata['control']['brake']),
                           float(step_metadata['control']['steer'])])
        return observation, action

    def sample(self, batch_size: int):

        # (obs, act)
        all_offline_samples = [defaultdict(list), defaultdict(list)]
        offline_samples = self._offline_buffers[3].sample(batch_size)

        # save each sub-batch
        for i, data in enumerate(offline_samples):
            all_offline_samples[i][3].extend(data)

        return all_offline_samples


if __name__ == '__main__':

    offline_buffer = OfflineBuffer(offline_buffer_hdf5='../dataset/encodings/encodings.hdf5',
                                   offline_buffer_json='../dataset/encodings/encodings.json')
    offline = offline_buffer.sample(batch_size=32)

