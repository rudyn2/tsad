import h5py
import random
from tqdm import tqdm
from utils.hdf5_saver import HDF5Saver
import argparse


class Splitter:

    def __init__(self, dataset_path: str, new_dataset_path: str):
        self.path = dataset_path
        self.new_path = new_dataset_path

    def split(self, percentage: float):
        assert 0 <= percentage <= 1

        with h5py.File(self.path, "r") as f:
            run_keys = list(f.keys())
            new_keys = random.sample(run_keys, int(percentage * len(run_keys)))
            saver = HDF5Saver(288, 288, file_path_to_save=self.new_path)

            for k in tqdm(new_keys, "Creating new dataset"):
                run_id = k
                media_data = []
                for step in sorted(f[k].keys()):
                    media_data.append({
                        'timestamp': step,
                        'rgb': f[k][step]['rgb'],
                        'depth': f[k][step]['depth'],
                        'semantic': f[k][step]['semantic']
                    })
                saver.save_one_ego_run(run_id, media_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Splitter utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../dataset', type=str, help='Path to dataset folder')
    parser.add_argument('--ratio', required=True, type=float, help='Size ratio of new hdf5 dataset')
    parser.add_argument('--output-data', required=True, type=str, help='Path to output hdf5 dataset')
    args = parser.parse_args()

    s = Splitter(args.data, args.output_data)
    s.split(args.ratio)
