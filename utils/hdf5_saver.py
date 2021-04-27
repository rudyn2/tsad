import h5py
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
