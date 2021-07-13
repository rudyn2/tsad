from models.carlaDatasetEncodings import CarlaEncodingDataset
from models.TemporalEncoder import SequenceRNNEncoder
from models.ADEncoder import ADEncoder
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
                 carla_dataset: CarlaEncodingDataset,
                 visual_model: ADEncoder,
                 temp_model: SequenceRNNEncoder,
                 output_path: str):
        self._carla_dataset = carla_dataset
        self._temp_model = temp_model
        self._visual_model = visual_model
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._hdf5_saver = HDF5EncodingSaver(output_path)

    def encode(self):
        data = defaultdict(dict)
        hidden = self._temp_model.init_hidden(batch_size=1, device=self._device)
        for idx in tqdm(range(len(self._carla_dataset)), "Encoding..."):
            x, act, speed, ts = self._carla_dataset[idx]
            x = x.to(self._device).unsqueeze(0)  # x: (1, 4, 224, 288)
            act = act.to(self._device).unsqueeze(0)  # act: (1, 3)
            speed = speed.to(self._device).unsqueeze(0)  # speed: (1, 3)

            # generate the visual and temporal encodings
            visual_encoding = self._visual_model.encode(x)  # visual_encoding: (1, 512, 4, 4)
            _, hidden = self._temp_model.encode(visual_encoding.unsqueeze(0), act, speed, hidden=hidden)
            temporal_encoding = hidden[0].reshape(-1, 4, 4).unsqueeze(0)

            # concatenate
            encoding = torch.cat([visual_encoding, temporal_encoding], dim=1).squeeze(0)
            data[ts[0]][ts[1]] = encoding.detach().cpu().numpy()

        for run_id, data in tqdm(data.items(), "Saving..."):
            self._hdf5_saver.write_batch(run_id, data)
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
    dataset = CarlaEncodingDataset(hdf5_path=args.data + '.hdf5',
                                   json_path=args.data + '.json')
    extractor = Encoder(carla_dataset=dataset,
                        visual_model=visual_model,
                        temp_model=temp_model,
                        output_path=args.output)
    extractor.encode()
