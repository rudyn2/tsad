import argparse
import json

import h5py
import numpy as np
from tqdm import tqdm
from hdf5_saver import HDF5Saver
from json_saver import JsonSaver


class Filter:
    def __init__(self, hdf5_path: str, json_path: str, output_hdf5_path: str, output_json_path: str, speed_thresh: float):
        self.hdf5_path = hdf5_path
        self.json_path = json_path
        self.output_hdf5_path = output_hdf5_path
        self.output_json_path = output_json_path

        self.threshold = speed_thresh
        self.min_steps = 200
        self.hdf5_saver = HDF5Saver(288, 288, output_hdf5_path)
        self.json_saver = JsonSaver(output_json_path)
        self.json_data = self.read_json()

    def read_json(self):
        with open(self.json_path, "r+") as file:
            data = json.load(file)
        return data

    def format_run(self, file, run_key: str):
        run_media = []
        run_info = []
        for step in file[run_key].keys():
            media_step_dict = {
                'timestamp': step,
                'rgb': np.array(file[run_key][step]['rgb']),
                'semantic': np.array(file[run_key][step]['semantic']),
                'depth': np.array(file[run_key][step]['depth'])
            }
            info_step_dict = {
                'timestamp': step,
                'metadata': self.json_data[run_key][step]
            }
            run_media.append(media_step_dict)
            run_info.append(info_step_dict)

        run_media.sort(key=lambda x: x['timestamp'])
        return run_media, run_info

    def filter(self):

        with h5py.File(self.hdf5_path, "r") as f:
            run_keys = list(f.keys())
            total_saved_runs = 0
            too_short_runs = 0
            wrong_orientation_runs = 0
            incomplete_metadata = 0

            for run in tqdm(run_keys, "Processing "):
                counter = 0
                good_orientation = True

                if len(f[run]) < self.min_steps:
                    too_short_runs += 1
                    continue

                # process every step and check that at any step lane orientation is less than 90 degrees
                try:
                    for step in f[run].keys():
                        lane_orientation = self.json_data[run][step]["lane_orientation"]
                        hlc = self.json_data[run][step]['command']
                        if lane_orientation > 90 and hlc not in ["LEFT", "RIGHT"]:
                            good_orientation = False
                        if self.json_data[run][step]["speed"] <= 0.00001 and self.json_data[run][step]["tl_state"] == "Green":
                            counter += 1
                except KeyError:
                    incomplete_metadata += 1
                    continue

                wrong_orientation_runs += 1 if not good_orientation else 0
                if not good_orientation:
                    print(run, " discarded because one or more steps have an lane orientation greater than 90 degrees")

                if counter / len(f[run]) < self.threshold and good_orientation:
                    # save run data
                    media_data, info_data = self.format_run(f, run)
                    self.hdf5_saver.save_one_ego_run(run, media_data, verbose=False)
                    self.json_saver.save_one_ego_run(info_data, run)
                    total_saved_runs += 1

        print(f"Total saved runs: {total_saved_runs}/{len(run_keys)} ({100 * (total_saved_runs / len(run_keys)):.2f}%)")
        print(f"# too short runs discarded: {too_short_runs}")
        print(f"# wrong orientation runs discarded: {wrong_orientation_runs}")
        print(f"# incomplete metadata runs discarded: {incomplete_metadata}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Utility to filter an hdf5 dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, help='path to hdf5 file')
    parser.add_argument('-m', '--metadata', required=True, type=str, help='path to json file')
    parser.add_argument('-o', '--output', default='filtered.hdf5', type=str, help='path of hdf5 file to be created')
    parser.add_argument('-om', '--output-metadata', default='filtered.json', type=str,
                        help='path of json file to be created')
    parser.add_argument('-p', '--percentage', required=True, type=float,
                        help='Ratio of null speed steps required to filter out one episode')

    args = parser.parse_args()

    f = Filter(args.input, args.metadata, args.output, args.output_metadata, args.percentage)
    f.filter()
