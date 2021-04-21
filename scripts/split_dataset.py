import h5py
import random
from tqdm import tqdm


class HDF5Saver:
    def __init__(self, sensor_width, sensor_height, file_path_to_save="data/carla_dataset.hdf5"):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

        self.file = h5py.File(file_path_to_save, "a")
        # Storing metadata
        self.file.attrs['sensor_width'] = sensor_width
        self.file.attrs['sensor_height'] = sensor_height
        self.file.attrs['simulation_synchronization_type'] = "syncd"

    def save_one_ego_run(self, run_id: str, media_data: list, verbose: bool = False):
        # if a group already exits override its content
        if run_id in self.file.keys():
            del self.file[run_id]

        ego_run_group = self.file.create_group(run_id)

        step_iterator = tqdm(media_data, "Saving images ") if verbose else media_data
        for frame_dict in step_iterator:
            # one frame dict contains rgb, depth and semantic information
            timestamp = str(frame_dict["timestamp"])
            ego_run_timestamp_group = ego_run_group.create_group(timestamp)
            ego_run_timestamp_group.create_dataset('rgb', data=frame_dict['rgb'], compression='gzip')
            ego_run_timestamp_group.create_dataset('depth', data=frame_dict['depth'], compression='gzip')
            ego_run_timestamp_group.create_dataset('semantic', data=frame_dict['semantic'], compression='gzip')

    def close_hdf5(self):
        self.file.close()


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
    s = Splitter('/home/rudy/Documents/carla-dataset-runner/data/sample6/sample6.hdf5',
                 'small.hdf5')
    s.split(0.5)
