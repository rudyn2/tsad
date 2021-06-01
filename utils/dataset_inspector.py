"""
Tool to inspect a dataset and report some useful statistics. For example:

* Total number of episodes
* Total number of frames
* Total number of frames with red lights
* Total number of frames per high level command
* Average episode length
"""
import h5py
import json
from termcolor import colored
from collections import defaultdict
import numpy as np
from datetime import timedelta
import argparse


class Inspector(object):

    def __init__(self, path: str):
        """
        Provided path should point to the dataset name. Then, the files with that name and .hdf5 and .json extensions
        will be searched.
        """
        self.path = path
        self.metadata = None

        # init functions
        self._load_metadata()

    def _load_metadata(self):
        with open(self.path + ".json", "r") as f:
            self.metadata = json.load(f)

    def inspect(self):

        image_dataset = h5py.File(self.path + ".hdf5", "r")

        # look for coherence between hdf5 and json keys
        image_run_ids = list(image_dataset.keys())
        metadata_run_ids = list(self.metadata.keys())

        valid_keys = self.check_sync(image_run_ids, metadata_run_ids)
        count_metrics_command, count_metrics_traffic_light = defaultdict(int), defaultdict(int)
        run_lengths = []
        total_time = 0
        for run_id in valid_keys:
            timestamps = list(self.metadata[run_id].keys())
            total_time += 5 * len(timestamps) / 20  # for every frame, others 5 were skipped. simulation at 20 fps.
            run_lengths.append(len(timestamps))
            for timestamp in timestamps:
                step_metadata = self.metadata[run_id][timestamp]
                count_metrics_command[step_metadata['command']] += 1
                count_metrics_traffic_light[step_metadata['tl_state']] += 1

        average_episode_length, std_episode_length = np.mean(run_lengths), np.std(run_lengths)
        print(colored("GENERAL METRICS", "green"))
        print(colored(f"Total episodes: {len(valid_keys):.0f}", "cyan"))
        print(colored(f"Total frames: {np.sum(run_lengths):.0f}", "cyan"))
        print(colored(f"Mean/Std episode length: {average_episode_length:.0f}/{std_episode_length:.0f}", "cyan"))
        print(colored(f"Estimated driving time: {timedelta(seconds=total_time)}", "cyan"))

        self.print_metrics(count_metrics_command, "COMMAND COUNT")
        self.print_metrics(count_metrics_traffic_light, "TRAFFIC LIGHT COUNT")

    @staticmethod
    def print_metrics(metrics: dict, header: str):
        print(colored(header, "green"))
        for k, v in metrics.items():
            print(colored(f"{k}: {v}", "cyan"))

    @staticmethod
    def check_sync(image_run_ids, metadata_run_ids):
        not_found_keys = set(image_run_ids) - set(metadata_run_ids)
        if len(not_found_keys) > 0:
            print(colored(f"{len(not_found_keys)} run ids were not found in json file", "red"))
            for run_id in not_found_keys:
                print(run_id)
            valid_keys = set(image_run_ids).intersection(set(metadata_run_ids))
        else:
            overhead = set(metadata_run_ids) - set(image_run_ids)
            valid_keys = set(image_run_ids)
            if len(overhead) > 0:
                print(colored(f"All run ids were found in json file but there are also {len(overhead)} "
                              f"runs that aren't contained in the hdf5 file.", "yellow"))
            else:
                print(colored("All run ids were found in json file, they are synced.", "green"))
        return list(valid_keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspector utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_path', default='../dataset', type=str,
                        help='Path to dataset (just name, without extension')
    args = parser.parse_args()
    i = Inspector(args.dataset_path)
    i.inspect()
