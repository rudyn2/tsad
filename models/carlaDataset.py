import json
from pathlib import Path

import h5py
from torch.utils import data


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, recursive=True, load_data=False, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.load_data = load_data

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.hdf5'))
            data = sorted(p.glob('**/*.json'))
        else:
            files = sorted(p.glob('*.hdf5'))
            data = sorted(p.glob('*.json'))

        if len(files) < 1 or len(data) != len(files):
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp, json_fp in zip(files, data):
            self._add_data_infos(str(h5dataset_fp.resolve()), str(json_fp.resolve()))

    def __getitem__(self, index):
        # get data
        imgs, data = self.get_data(index)

        x = np.concatenate((imgs['rgb'], imgs['depth'][:, :, np.newaxis]), axis=2)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # # get label
        y = imgs['semantic']
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.data_info)

    def _add_data_infos(self, file_path, json_path):
        with h5py.File(file_path, 'r') as h5_file, open(json_path) as json_file:
            control = json.load(json_file)
            # Walk through all groups, extracting datasets
            for ep_name, episode in h5_file.items():
                for tname, timestamp in episode.items():
                    shapes = []
                    datas = {}
                    for dname, data in timestamp.items():
                        shapes.append(data[()].shape)
                        datas[dname] = data[()]

                    # if data is not loaded its cache index is -1
                    idx = -1
                    if self.load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(datas, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'episode': ep_name, 'timestamp': tname, 'shape': shapes,
                         'data': control[ep_name][tname], 'cache_idx': idx})

    def _load_data(self, file_path, json_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, 'r') as h5_file, open(json_path) as json_file:
            data = json.load(json_file)
            # Walk through all groups, extracting datasets
            for ep_name, episode in h5_file.items():
                for tname, timestamp in episode.items():
                    shapes = []
                    datas = {}
                    for dname, data in data.items():
                        shapes.append(data[()].shape)
                        datas[dname] = data[()]

                        # add data to the data cache and retrieve
                        # the cache index
                        idx = self._add_to_cache(datas, file_path)

                        # find the beginning index of the hdf5 file we are looking for
                        file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                        # the data info should have the same index since we loaded it in the same way
                        self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'episode': di['episode'], 'timestamp': di['timestamp'],
                               'shape': di['shape'], 'data': di['data'], 'cache_idx': -1} if di['file_path'] ==
                                                                                             removal_keys[0] else di for
                              di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data(self, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        timestamp = self.data_info[i]
        cache_idx = timestamp['cache_idx']
        fp = timestamp['file_path']
        if self.load_data:
            if fp not in self.data_cache:
                self._load_data(fp)
                # get new cache_idx assigned by _load_data_info
                cache_idx = self.data_info[i]['cache_idx']
            return self.data_cache[fp][cache_idx], self.data_info[i]

        with h5py.File(fp, 'r') as f:
            datas = {}
            for dname, data in f[timestamp['episode']][timestamp['timestamp']].items():
                datas[dname] = data[()]
        return datas, timestamp


if __name__ == "__main__":
    path = 'dataset'
    # f = h5py.File(path, 'r')
    # print(f['run_000_morning']['depth']['1617979592423'])
    dataset =HDF5Dataset(path)
print(dataset[10])