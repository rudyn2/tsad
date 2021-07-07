from models.carlaEmbeddingDataset import CarlaOnlineEmbeddingDataset
from models.TemporalEncoder import VanillaRNNEncoder
from utils.json_saver import JsonSaver
import argparse
import torch
from collections import defaultdict
import h5py
from tqdm import tqdm


class HDF5EncodingSaver:

    def __init__(self, path: str):
        self._path = path

    def write_batch(self, run_id: str, data: dict):
        with h5py.File(self._path, mode='a') as f:
            run_group = f.create_group(run_id)
            for timestamp, encoding in data.items():
                run_group.create_dataset(timestamp, data=encoding)


class Encoder(object):
    """
    Given visual embeddings and a trained RNN model, this class generates a encoding composed of the concatenation of
    the visual embedding and the hidden state of the trained RNN model when it is used to calculate the next embedding.
    """

    def __init__(self,
                 embedding_dataset: CarlaOnlineEmbeddingDataset,
                 temp_model: VanillaRNNEncoder,
                 output_path: str):
        self._embedding_dataset = embedding_dataset
        self._temp_model = temp_model
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._hdf5_saver = HDF5EncodingSaver(output_path)

    def encode(self):
        data = defaultdict(dict)

        for idx in tqdm(range(len(self._embedding_dataset)), "Encoding..."):
            seq, act, speed, _, ts = self._embedding_dataset[idx]
            seq = seq.to(self._device).unsqueeze(0)  # seq: (1, 4, 512, 4, 4)
            act = act.to(self._device).unsqueeze(0)
            speed = speed.to(self._device).unsqueeze(0)
            temporal_encoding = self._temp_model(seq, act, speed)
            encoding = torch.cat([seq[:, -1, :, :], temporal_encoding], dim=1).squeeze(0)
            data[ts[0]][ts[1]] = encoding.detach().cpu().numpy()

        for run_id, data in tqdm(data.items(), "Saving..."):
            self._hdf5_saver.write_batch(run_id, data)
        print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the visual+temp encodings using the precalculated embeddings"
                                                 "and a RNN encoder ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../dataset/embeddings/clean_1', type=str, help='Path to dataset folder')
    parser.add_argument('--output', default='encodings.hdf5', type=str,
                        help='Output path of generated embeddings')
    parser.add_argument('--weights', required=True, type=str, help='Path to model weights')
    parser.add_argument('--device', default='cuda', type=str, help='Device [cuda, cpu]')
    args = parser.parse_args()

    # create model
    model = VanillaRNNEncoder(num_layers=4,
                              hidden_size=1024,
                              action__chn=256,
                              speed_chn=256,
                              bidirectional=True)
    model.load_state_dict(torch.load(args.weights))
    model.to(args.device)
    model.eval()

    # instantiate dataset
    dataset = CarlaOnlineEmbeddingDataset(embeddings_path=args.data + '.hdf5',
                                          json_path=args.data + '.json',
                                          provide_ts=True)
    extractor = Encoder(embedding_dataset=dataset, temp_model=model, output_path=args.output)
    extractor.encode()
