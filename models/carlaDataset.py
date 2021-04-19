from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import h5py
import os


class CarlaDataset(Dataset):

    def __init__(self, path: str):
        """
        Constructor method of CarlaDataset
        :param path: path to media and info file. expected to found files at {path}.hdf5 and {path}.json
        :type path: str
        """
        super(CarlaDataset, self).__init__()
        self.path = path
        assert os.path.exists(os.path.join(path, ".hdf5")), "We couldn't find .hdf5 file exists at specified path"
        assert os.path.exists(os.path.join(path, ".json")), "We couldn't find .json file exists at specified path"

    def __getitem__(self, index) -> T_co:
        pass



