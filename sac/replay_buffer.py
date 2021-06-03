import json
import random

import h5py
import numpy as np
import torch
from tqdm import tqdm
from sac.utils import calc_reward


class StateObj(object):
    def __init__(self, visual_embedding: torch.Tensor, speed: float, hlc: str):
        self.visual_embedding = visual_embedding
        self.speed = speed
        self.hlc = hlc


class ReplayMemoryFast:

    # first we define init method and initialize buffer size
    def __init__(self, memory_size):

        # max number of samples to store
        self.memory_size = memory_size

        self.experience = [None] * self.memory_size
        self.current_index: int = 0
        self.size = 0

    # next we define the function called store for storing the experiences
    def add(self, observation, action, reward, newobservation, not_done, not_done_no_max):

        # store the experience as a tuple (current state, action, reward, next state, is it a terminal state)
        self.experience[self.current_index] = (observation, action, reward, newobservation, not_done, not_done_no_max)
        self.current_index += 1

        self.size = min(self.size + 1, self.memory_size)

        # if the index is greater than  memory then we flush the index by subtrating it with memory size
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    # we define a function called sample for sampling the minibatch of experience
    def sample(self, batch_size: int) -> list:
        """
        Sample from buffer.
        :param batch_size: number of samples to collect.
        :type batch_size: int
        :return: a list with 6 lists (observation, action, reward, new_observation, not_done, not_done_no_max)
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
        :return: a list of tuples (observation, action, reward, new_observation, not_done, not_done_no_max)
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

    def __init__(self, online_memory_size: int, offline_buffer_hdf5: str, offline_buffer_json: str):
        self._offline_buffer_hdf5_path = offline_buffer_hdf5
        self._offline_buffer_json_path = offline_buffer_json
        self._online_buffer = ReplayMemoryFast(online_memory_size)
        self._offline_buffer = self._load()

    def _get_total_steps(self):
        total_steps = 0
        with h5py.File(self._offline_buffer_hdf5_path, "r") as f:
            for run_id in f.keys():
                total_steps += len(f[run_id])
        return total_steps

    def _load(self) -> ReplayMemoryFast:
        with open(self._offline_buffer_json_path, "r") as f:
            metadata = json.load(f)

        total_offline_steps = self._get_total_steps()
        offline_buffer = ReplayMemoryFast(memory_size=total_offline_steps)
        with h5py.File(self._offline_buffer_hdf5_path, "r") as f:
            for run_id in tqdm(list(f.keys())[:5], "Loading dataset"):
                episode_metadata = metadata[run_id]
                steps = list(f[run_id].keys())
                steps = sorted(steps)   # to be sure that they are in order
                offline_buffer.memory_size += len(steps)
                for idx in range(len(steps) - 1):
                    step, next_step = steps[idx], steps[idx + 1]
                    done = 1 if idx == len(steps) - 1 else 0
                    transition = self._get_transition(f[run_id], episode_metadata, step, next_step)
                    transition = *transition, done, done
                    offline_buffer.add(*transition)
        return offline_buffer

    def _get_transition(self, h5py_group: h5py.File, metadata_json: dict, step: str, next_step: str):

        visual, next_visual = np.array(h5py_group[step]), np.array(h5py_group[step])
        step_metadata, next_step_metadata = metadata_json[step], metadata_json[next_step]
        state = np.array([float(step_metadata['speed']),
                          float(self.HLC_TO_NUMBER[step_metadata['command']])])
        next_state = np.array([float(next_step_metadata['speed']),
                               float(self.HLC_TO_NUMBER[next_step_metadata['command']])])
        observation = dict(visual=visual, state=state)
        next_observation = dict(visual=next_visual, state=next_state)
        reward = calc_reward(step_metadata)          # tbd
        action = np.array([float(step_metadata['control']['steer']),
                           float(step_metadata['control']['throttle']),
                           float(step_metadata['control']['brake'])])
        return observation, action, reward, next_observation

    def sample(self, batch_size: int, offline: float):
        assert 0 <= offline <= 1, f"Offline relative size should be between 0 and 1, got: {offline}"
        online_batch_size = int(batch_size * (1 - offline))
        offline_batch_size = batch_size - online_batch_size

        offline_samples = self._offline_buffer.unpacked_sample(offline_batch_size)
        online_samples = self._online_buffer.unpacked_sample(online_batch_size)
        if len(online_samples) == 0:
            online_samples = self._offline_buffer.unpacked_sample(online_batch_size)

        return offline_samples, online_samples

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        pass

# class ReplayBuffer(object):
#     """Buffer to store environment transitions."""
#
#     def __init__(self, obs_shape, action_shape, capacity, device):
#         self.capacity = capacity
#         self.device = device
#
#         # the proprioceptive obs is stored as float32, pixels obs as uint8
#         obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
#
#         self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
#         self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
#         self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
#         self.rewards = np.empty((capacity, 1), dtype=np.float32)
#         self.not_dones = np.empty((capacity, 1), dtype=np.float32)
#         self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
#
#         self.idx = 0
#         self.last_save = 0
#         self.full = False
#
#     def __len__(self):
#         return self.capacity if self.full else self.idx
#
#     def add(self, obs, action, reward, next_obs, done, done_no_max):
#         np.copyto(self.obses[self.idx], obs)
#         np.copyto(self.actions[self.idx], action)
#         np.copyto(self.rewards[self.idx], reward)
#         np.copyto(self.next_obses[self.idx], next_obs)
#         np.copyto(self.not_dones[self.idx], not done)
#         np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
#
#         self.idx = (self.idx + 1) % self.capacity
#         self.full = self.full or self.idx == 0
#
#     def sample(self, batch_size):
#         idxs = np.random.randint(0,
#                                  self.capacity if self.full else self.idx,
#                                  size=batch_size)
#
#         obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
#         actions = torch.as_tensor(self.actions[idxs], device=self.device)
#         rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
#         next_obses = torch.as_tensor(self.next_obses[idxs],
#                                      device=self.device).float()
#         not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
#         not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
#                                            device=self.device)
#
#         return obses, actions, rewards, next_obses, not_dones, not_dones_no_max


if __name__ == '__main__':
    mixed_replay_buffer = MixedReplayBuffer(512,
                                            offline_buffer_hdf5='/tmp/embeddings_noflat.hdf5',
                                            offline_buffer_json='/tmp/carla_dataset_temp.json')
    sample = mixed_replay_buffer.sample(batch_size=512, offline=0.7)
