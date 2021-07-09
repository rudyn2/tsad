import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import sys

sys.path.append('..')
from models.carlaEmbeddingDataset import CarlaOnlineEmbeddingDataset, PadSequence, CarlaEmbeddingDataset
from models.TemporalEncoder import RNNEncoder, VanillaRNNEncoder
from models.ADEncoder import ADEncoder


def save_image(array, path, vmin=0, vmax=255):
    plt.imshow(array, vmin=vmin, vmax=vmax)
    plt.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predicted embeddings from temporal encoder",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--embeddings', default='../dataset/embeddings.hdf5', type=str, help='Path to embeddings hdf5')
    parser.add_argument('--metadata', default='../dataset/carla_dataset_repaired.json', type=str,
                        help='Path to json file')
    parser.add_argument('--dataset', default='online', type=str, help='Type of dataset. online: all the dataset will'
                                                                      'be loaded into memory before training. '
                                                                      'offline: the embeddings'
                                                                      'will be loaded lazily')

    parser.add_argument('--vis-weights', required=True, type=str, help='Path to visual encoder weights')
    parser.add_argument('--backbone', default='efficientnet-b5', type=str, help="Model's backbone")

    parser.add_argument('--img-folder', default='images', type=str, help="Folder where to save images")
    parser.add_argument('--nb-images', default=100, type=int, help="Minimum number of images to recreate")

    parser.add_argument('--temp-weights', required=True, type=str, help='Path to temporal encoder weights')
    parser.add_argument('--hidden-size', default=256, type=int, help='LSTM hidden size')
    parser.add_argument('--num-layers', default=2, type=int, help='LSTM number of hidden layers')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')

    parser.add_argument('--action-channels', default=128, type=int, help='Number of channels in action codification')
    parser.add_argument('--speed-channels', default=128, type=int, help='Number of channels in speed codification')
    parser.add_argument('--rnn-model', default='vanilla', type=str, help='Which rnn model use: "vanilla" or "convol"')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional LSTM or not.')
    args = parser.parse_args()

    if not os.path.exists(args.img_folder):
        os.makedirs(args.img_folder)

    device = args.device
    if args.dataset == 'online':
        dataset = CarlaOnlineEmbeddingDataset(embeddings_path=args.embeddings, json_path=args.metadata)
    else:
        dataset = CarlaEmbeddingDataset(embeddings_path=args.embeddings, json_path=args.metadata)

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=PadSequence())

    if args.rnn_model == "vanilla":
        temp_model = VanillaRNNEncoder(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            action__chn=args.action_channels,
            speed_chn=args.speed_channels,
            bidirectional=args.bidirectional,
        )
    elif args.rnn_model == "convol":
        temp_model = RNNEncoder(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            action__chn=args.action_channels,
            speed_chn=args.speed_channels
        )
    else:
        raise NotImplementedError('Model not implemented')

    temp_model.load_state_dict(torch.load(args.temp_weights))
    temp_model.to(args.device)
    temp_model.eval()

    # create model
    vis_model = ADEncoder(backbone=args.backbone)
    vis_model.load_state_dict(torch.load(args.vis_weights))
    vis_model.to(args.device)
    vis_model.eval()

    for i, (embeddings, embeddings_length, actions, speeds, embeddings_label) in enumerate(loader):
        embeddings, embeddings_label, actions, speeds = embeddings.to(args.device), embeddings_label.to(
            args.device), actions.to(
            args.device), speeds.to(args.device)
        pred_embedding = temp_model(embeddings, actions, speeds, embeddings_length)

        output_expected = vis_model.decode(embeddings_label)
        segmentation = output_expected['segmentation'].argmax(dim=1)

        output_predicted = vis_model.decode(pred_embedding)
        pred_segmentation = output_predicted['segmentation'].argmax(dim=1)

        vmin = pred_segmentation.min()
        vmax = pred_segmentation.max()

        for j in range(pred_segmentation.shape[0]):
            pred_seg = pred_segmentation[j].detach().cpu().numpy()
            seg = segmentation[j].detach().cpu().numpy()
            save_image(pred_seg, '{}/{}-{}_pred.png'.format(args.img_folder, i, j), vmin=vmin, vmax=vmax)
            save_image(seg, '{}/{}-{}_real.png'.format(args.img_folder, i, j), vmin=vmin, vmax=vmax)

        if i >= args.nb_images // args.batch_size + 1:
            break
