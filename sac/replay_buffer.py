import json

import h5py
import numpy as np
from tqdm import tqdm
from sac.utils import calc_reward
from typing import Tuple, Dict
from collections import defaultdict
from termcolor import colored


class ReplayMemoryFast:

    # first we define init method and initialize buffer size
    def __init__(self, memory_size: int):

        # max number of samples to store
        self.memory_size: int = memory_size

        self.experience: list = [None] * self.memory_size
        self.current_index: int = 0
        self.size = 0

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


class MixedReplayBuffer(object):
    HLC_TO_NUMBER = {
        'RIGHT': 0,
        'LEFT': 1,
        'STRAIGHT': 2,
        'LANEFOLLOW': 3
    }

    def __init__(self,
                 online_memory_size: int,
                 reward_weights: tuple,
                 offline_buffer_hdf5: str = None,
                 offline_buffer_json: str = None):
        self._offline_buffer_hdf5_path = offline_buffer_hdf5
        self._offline_buffer_json_path = offline_buffer_json

        # create buffer per each high level command
        self._online_buffers = {
            0: ReplayMemoryFast(online_memory_size),    # RIGHT
            1: ReplayMemoryFast(online_memory_size),    # LEFT
            2: ReplayMemoryFast(online_memory_size),    # STRAIGHT
            3: ReplayMemoryFast(online_memory_size)     # LANE FOLLOW
        }
        self.reward_weights = reward_weights

        self._offline_buffers = None
        if self._offline_buffer_json_path and self._offline_buffer_hdf5_path:
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
                    step, next_step = steps[idx], steps[idx + 1]
                    not_done = 0 if idx == len(steps) - 1 else 1
                    transition = self._get_transition(f[run_id], episode_metadata, step, next_step)
                    transition = *transition, not_done
                    # add to some buffer depending on the HLC command of the current observation
                    offline_buffers[transition[0]['hlc']].add(*transition)
        return offline_buffers

    def _get_transition(self, h5py_group: h5py.File, metadata_json: dict, step: str, next_step: str) \
            -> Tuple[Dict[str, np.ndarray], np.ndarray, float, Dict[str, np.ndarray]]:

        encoding, next_encoding = np.array(h5py_group[step]), np.array(h5py_group[next_step])
        step_metadata, next_step_metadata = metadata_json[step], metadata_json[next_step]
        step_speed = np.array([float(step_metadata['speed_x']), float(step_metadata['speed_y']),
                               float(step_metadata['speed_z'])])
        step_command = int(self.HLC_TO_NUMBER[step_metadata['command']])
        next_step_speed = np.array([float(next_step_metadata['speed_x']), float(next_step_metadata['speed_y']),
                                    float(next_step_metadata['speed_z'])])
        next_step_command = int(self.HLC_TO_NUMBER[next_step_metadata['command']])
        observation = dict(encoding=encoding, speed=step_speed, hlc=step_command)
        next_observation = dict(encoding=next_encoding, speed=next_step_speed, hlc=next_step_command)
        reward = calc_reward(step_metadata, reward_weights=self.reward_weights)
        action = np.array([float(step_metadata['control']['throttle']),
                           float(step_metadata['control']['brake']),
                           float(step_metadata['control']['steer'])])
        return observation, action, reward, next_observation

    def sample(self, batch_size: int, offline: float = 0.25):
        """
        Returns a sample of experiences. <offline> should be the relative size of the
        offline samples. The sample is a tuple of 5 elements, one per each data type (obs, act, rew, next_obs, not_done),
        and is organized as follows:
            (
                {
                    0: [...],   # high level commands
                    1: [...],
                    2: [...],
                    3: [...]
                }, # end of the observations
                ...
            )
        """
        assert 0 <= offline <= 1, f"Offline relative size should be between 0 and 1, got: {offline}"
        online_batch_size = (int(batch_size * (1 - offline)))
        offline_batch_size = (batch_size - online_batch_size)

        # (obs, act, rew, next_obs, not_done)
        all_online_samples, all_offline_samples = [defaultdict(list) for _ in range(5)], \
                                                  [defaultdict(list) for _ in range(5)]
        offline_samples = []
        if self._offline_buffers:
            offline_samples = self._offline_buffers[3].sample(offline_batch_size)
        if len(offline_samples) == 0:  # then either the buffer doesn't exists or doesn't have enough samples
            # so, we try to pull all of them from the online buffer
            online_samples = self._online_buffers[3].sample(batch_size)
        else:  # if we have samples from the offline buffer, then we just get the missing ones
            online_samples = self._online_buffers[3].sample(online_batch_size)
            if len(online_samples) == 0:  # then we don't have enough samples in the online buffer
                online_samples = self._offline_buffers[3].sample(online_batch_size)

        # save each sub-batch
        for i, data in enumerate(offline_samples):
            all_offline_samples[i][3].extend(data)
        for i, data in enumerate(online_samples):
            all_online_samples[i][3].extend(data)

        return all_online_samples, all_offline_samples

    def add(self, obs, action, reward, next_obs, done):
        """
        Add a transition to the online buffer.
        """
        self._online_buffers[int(obs['hlc'])].add(obs, action, reward, next_obs, done)

    def log_status(self):
        s = ""
        if self._offline_buffers is not None:
            s += "\nOffline buffers:"
            for k in self._offline_buffers.keys():
                s += f"{k}: {len(self._offline_buffers[k])} samples\n"
        else:
            s += "\nNo offline buffer"
        s += "\nOnline buffers:\n"
        for k in self._online_buffers.keys():
            s += f"{k}: {len(self._online_buffers[k])} samples\n"
        return s


if __name__ == '__main__':

    mixed_replay_buffer = MixedReplayBuffer(512,
                                            reward_weights=(0.3, 0.3, 0.3),
                                            offline_buffer_hdf5='../dataset/encodings/encodings.hdf5',
                                            offline_buffer_json='../dataset/encodings/encodings.json')
    offline, online = mixed_replay_buffer.sample(batch_size=32, offline=0.25)

