"""
Tool to join hdf5 datasets.
"""
import h5py
import json
import argparse
from hdf5_saver import HDF5Saver
from json_saver import JsonSaver
from glob import glob
import numpy as np
from tqdm import tqdm


class Merger(object):

    def __init__(self, folder_path: str, output_path: str):
        self.path = folder_path
        self.metadata = {}
        self.hdf5_saver = HDF5Saver(288, 288, file_path_to_save=output_path + ".hdf5")
        self.json_saver = JsonSaver(path=output_path + ".json")
        self._load_metadata()
        self.total_saved = 0
        self.total_discarded = 0

    def _load_metadata(self):
        json_files = glob(self.path + "*.json")
        for json_file in json_files:
            with open(json_file, "r") as f:
                json_metadata = json.load(f)
            self.metadata.update(json_metadata)

    def merge(self):
        # find all hdf5 files in provided path
        hdf5_files = glob(self.path + "*.hdf5")
        print("The following files will be merged: ")
        for hdf5_file in hdf5_files:
            print(hdf5_file)
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, "r") as f:
                self.process_hdf5_file(f)

    def process_hdf5_file(self, file: h5py.File):
        """
        Save all episodes of provided file
        """
        for run_id in tqdm(file.keys(), "Processing..."):
            if run_id in self.metadata.keys():
                image_ts = list(file[run_id].keys())
                meta_ts = list(self.metadata[run_id].keys())
                if set(image_ts) == set(meta_ts):

                    media_data = [{
                        "timestamp": ts,
                        "rgb": np.array(file[run_id][ts]["rgb"]),
                        "depth": np.array(file[run_id][ts]["depth"]),
                        "semantic": np.array(file[run_id][ts]["semantic"])
                    } for ts in image_ts]
                    info_data = [{
                        "timestamp": ts,
                        "metadata": self.metadata[run_id][ts]
                    } for ts in meta_ts]

                    self.hdf5_saver.save_one_ego_run(run_id, media_data, verbose=False)
                    self.json_saver.save_one_ego_run(run_id=run_id, info_data=info_data)
                    self.total_saved += 1
                else:
                    self.total_discarded += 1
            else:
                self.total_discarded += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merger utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('datasets_folder', default='../dataset', type=str,
                        help='Path to dataset (just name, without extension')
    parser.add_argument('--output', type=str, default="merge", help="Output path name (without extensions)")
    args = parser.parse_args()
    i = Merger(args.datasets_folder, args.output)
    i.merge()
