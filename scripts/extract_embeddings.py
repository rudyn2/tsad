import argparse

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

sys.path.append('..')
sys.path.append('.')

from models.ADEncoder import ADEncoder
from models.carlaDatasetSimple import CarlaDatasetSimple
from models.carlaDatasetTransform import CarlaDatasetTransform


class SubDataset(Dataset):
    def __init__(self, original_dataset: CarlaDatasetTransform, indexes: list):
        self._dataset = original_dataset
        self._indexes = indexes

    def __getitem__(self, item):
        new_idx = self._indexes[item]
        return new_idx, self._dataset[new_idx][0]

    def __len__(self):
        return len(self._indexes)


class HDF5EmbeddingSaver:

    def __init__(self, path: str):
        self._path = path

    def write_batch(self, run_id: str, data: dict):
        with h5py.File(self._path, mode='a') as f:
            run_group = f.create_group(run_id)
            for timestamp, embedding in data.items():
                run_group.create_dataset(timestamp, data=embedding)


class Extractor:

    def __init__(self, model: ADEncoder, dataset: CarlaDatasetTransform, output_path: str, device: str = 'cuda'):
        self._model = model
        self._dataset = dataset
        self.__device = device
        self._output_path = output_path
        self._hdf5_handler = HDF5EmbeddingSaver(output_path)

    def extract(self):
        metadata = self._dataset.get_metadata()
        timestamps_map = self._dataset.get_timestamps()

        for run_id in tqdm(metadata.keys(), "Encoding..."):
            timestamps = metadata[run_id].keys()
            try:
                if len(timestamps) == 0:
                    print(f"Couldn't find keys for run id: {run_id}. Skipping...")
                idxs = [timestamps_map.index(t) for t in timestamps]
                data = self.extract_single(timestamps, idxs)
                self._hdf5_handler.write_batch(run_id, data)
            except ValueError as e:
                print(f"Couldn't process run with id: {run_id}")

    def extract_batch(self, timestamps, indexes):
        """
        Extract the embedding batch-by-batch
        """
        # TODO: finish this implementation and test improvement
        idx2timestamp = dict(zip(indexes, timestamps))
        idx2timestamp_v = np.vectorize(lambda e: int(idx2timestamp[e]))

        temp_dataset = SubDataset(self._dataset, indexes)
        temp_loader = DataLoader(temp_dataset, batch_size=4)

        embeddings = []
        items = []
        for item, x in temp_loader:
            x = x.to(self.__device)
            embedding = self._model.encode(x)
            item = item.detach().cpu().numpy()
            item = idx2timestamp_v(item)
            items.append(item)
            # embeddings.append(torch.flatten(embedding, start_dim=1).detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())

    def extract_single(self, timestamps, indexes):
        """
        Extract the embedding one-by-one
        """
        data = {}
        for idx, t in zip(indexes, timestamps):
            x = self._dataset[idx][0].unsqueeze(0).to(self.__device)
            embedding = self._model.encode(x)
            # embedding = torch.flatten(embedding, start_dim=0).detach().cpu().numpy()
            embedding = embedding.squeeze(dim=0).detach().cpu().numpy()
            data[t] = embedding
        return data

    def write_batch(self, run_id: str, data: dict):
        self._hdf5_handler.write_batch(run_id, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the encodings of provided dataset using some model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../dataset', type=str, help='Path to dataset folder')
    parser.add_argument('--output', default='embeddings.hdf5', type=str,
                        help='Output path of generated embeddings')
    parser.add_argument('--backbone', default='efficientnet-b5', type=str, help="Model's backbone")
    parser.add_argument('--weights', required=True, type=str, help='Path to model weights')
    parser.add_argument('--device', default='cuda', type=str, help='Device [cuda, cpu]')
    args = parser.parse_args()

    # create model
    model = ADEncoder(backbone=args.backbone)
    model.load_state_dict(torch.load(args.weights))
    model.to(args.device)
    model.eval()

    # instantiate dataset
    dataset = CarlaDatasetTransform(path=args.data, prob=0.0)
    extractor = Extractor(model=model, dataset=dataset, output_path=args.output)
    extractor.extract()
