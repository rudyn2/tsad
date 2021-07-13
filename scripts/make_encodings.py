from models.carlaDatasetEncodings import CarlaEncodingDataset
from models.TemporalEncoder import SequenceRNNEncoder
from models.ADEncoder import ADEncoder
import argparse
import torch
from collections import defaultdict
import h5py
import json
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np


class HDF5EncodingSaver:

    def __init__(self, path: str):
        self._path = path

    def write_batch(self, run_id: str, data: dict):
        with h5py.File(self._path, mode='a') as f:
            run_group = f.create_group(run_id)
            for timestamp, encoding in data.items():
                run_group.create_dataset(timestamp, data=encoding)

    def write_ts(self, run_id: str, step: str, data):
        with h5py.File(self._path, mode='a') as f:
            if run_id not in f.keys():
                run_group = f.create_group(run_id)
            else:
                run_group = f[run_id]
            if step not in f[run_id].keys():
                run_group.create_dataset(step, data=data)


class Encoder(object):
    """
    Given visual embeddings and a trained RNN model, this class generates a encoding composed of the concatenation of
    the visual embedding and the hidden state of the trained RNN model when it is used to calculate the next embedding.
    """

    def __init__(self,
                 data: str,
                 visual_model: ADEncoder,
                 temp_model: SequenceRNNEncoder,
                 output_path: str):
        self._hdf5_path = data + '.hdf5'
        self._json_path = data + '.json'
        self._temp_model = temp_model
        self._visual_model = visual_model
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._hdf5_saver = HDF5EncodingSaver(output_path)

        self.to_tensor = T.ToTensor()

    def encode(self):

        with open(self._json_path, "r") as f:
            all_metadata = json.load(f)

        with h5py.File(self._hdf5_path, "r") as f:
            pbar = tqdm(iterable=f.keys(), total=len(f.keys()), desc='')
            for run_id in pbar:
                pbar.desc = f"Encoding {run_id}"

                hidden = self._temp_model.init_hidden(batch_size=1, device=self._device)
                for ts in f[run_id].keys():
                    rgb = np.array(f[run_id][ts]['rgb'])
                    depth = np.array(f[run_id][ts]['depth'])

                    # process and stack
                    rgb_transformed = self.to_tensor(rgb)
                    depth_transformed = torch.tensor(depth).unsqueeze(0) / 1000
                    input_ = torch.cat([depth_transformed, rgb_transformed]).float()
                    input_ = input_[:, 64:, :]
                    input_ = input_.unsqueeze(0).to(self._device)

                    metadata = all_metadata[run_id][ts]
                    action = torch.tensor(
                        [metadata['control']['steer'], metadata['control']['throttle'], metadata['control']['brake']],
                        device=self._device).unsqueeze(0)
                    speed = torch.tensor([metadata['speed_x'], metadata['speed_y'], metadata['speed_y']],
                                         device=self._device).unsqueeze(0)

                    # generate the visual and temporal encodings
                    visual_encoding = self._visual_model.encode(input_)  # visual_encoding: (1, 512, 4, 4)
                    _, hidden = self._temp_model.encode(visual_encoding.unsqueeze(0), action, speed, hidden=hidden)
                    temporal_encoding = hidden[0].reshape(-1, 4, 4).unsqueeze(0)

                    # concatenate
                    encoding = torch.cat([visual_encoding, temporal_encoding], dim=1).squeeze(0)
                    self._hdf5_saver.write_ts(run_id, ts, data=encoding.detach().cpu().numpy())
                pbar.update(1)
        print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the visual+temp encodings using the precalculated embeddings"
                                                 "and a RNN encoder ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to dataset folder')
    parser.add_argument('--output', default='encodings.hdf5', type=str,
                        help='Output path of generated embeddings')
    parser.add_argument('--temp-weights', required=True, type=str, help='Path to temporal model weights')
    parser.add_argument('--visual-weights', required=True, type=str, help='Path to visual model weights')
    parser.add_argument('--device', default='cuda', type=str, help='Device [cuda, cpu]')
    args = parser.parse_args()

    # create models
    temp_model = SequenceRNNEncoder(num_layers=2,
                                    hidden_size=1024,
                                    action__chn=1024,
                                    speed_chn=1024,
                                    bidirectional=True)
    temp_model.load_state_dict(torch.load(args.temp_weights))
    temp_model.to(args.device)
    temp_model.eval()

    visual_model = ADEncoder(backbone='mobilenetv3_small_075', use_timm=True)
    visual_model.load_state_dict(torch.load(args.visual_weights))
    visual_model.to(args.device)
    visual_model.eval()

    # instantiate dataset
    extractor = Encoder(data=args.data,
                        visual_model=visual_model,
                        temp_model=temp_model,
                        output_path=args.output)
    extractor.encode()
