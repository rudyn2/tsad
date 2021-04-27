"""
If some episode is corrupted inside the hdf5 file, h5py would raise an exception when it's trying
to read it. The exception looks like:

'Unable to open object (bad object header version number)'

Because of the corrupted information, h5py can't delete that specific episode. Then, this utility copies every
episode into a new h5py file discarding any episode that raises an exception when trying to read it.
"""
import h5py
from utils.hdf5_saver import HDF5Saver
import numpy as np
from tqdm import tqdm
import argparse


class WrongSemanticSegmentation(Exception):
    def __init__(self):
        pass


class H5DFRepair:

    def __init__(self, path: str, new_path: str):
        self.path = path
        self.new_path = new_path
        self.new_hdf5 = HDF5Saver(288, 288, new_path)

    def repair(self):
        with h5py.File(self.path, "r+") as f:
            total_t = 0
            total_discarded = 0
            for k in tqdm(f.keys(), "Repairing "):
                try:
                    total_t += len(f[k].keys())
                    media_data = []
                    for t in f[k].keys():
                        semantic = np.array(f[k][t]['semantic'])
                        if np.min(semantic) < 0 or np.max(semantic) > 22:
                            raise WrongSemanticSegmentation

                        media_data.append({
                            'timestamp': str(t),
                            'rgb': np.array(f[k][t]['rgb']),
                            'depth': np.array(f[k][t]['depth']),
                            'semantic': semantic
                        })
                    media_data.sort(key=lambda x: int(x['timestamp']))
                    self.new_hdf5.save_one_ego_run(run_id=str(k), media_data=media_data)

                except KeyError as e:
                    total_discarded += 1
                    # print(f"Error detected at run: {k}")
                    continue

                except WrongSemanticSegmentation as e:
                    total_discarded += 1
                    # print(f"Wrong semantic segmentation detected at run: {k}")
                    continue
            print(f"Total runs discarded: {total_discarded}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Repair HDF5 Utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, type=str, help='Path to HDF5 file.')
    parser.add_argument('--output', required=True, type=str, help='Path to repaired HDF5 file.')
    args = parser.parse_args()

    repairman = H5DFRepair(args.input, args.output)
    repairman.repair()
