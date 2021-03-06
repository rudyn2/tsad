import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import sys

sys.path.append('..')
from models.carlaEmbeddingDataset import CarlaOnlineEmbeddingDataset, PadSequence, CarlaEmbeddingDataset
from models.TemporalEncoder import RNNEncoder, VanillaRNNEncoder, SequenceRNNEncoder
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

    parser.add_argument('--use-sequence', action='store_true', help='Whether to use embeddings [0:n] to predict [1:n+1] or just n+1.')
    parser.add_argument('--force-timm', action='store_true', help='Whether to use timm library or not, only used when backbone is efficientnet.')
    args = parser.parse_args()

    save_imgs = False
    plot_emb = True

    if not os.path.exists(args.img_folder):
        os.makedirs(args.img_folder)

    device = args.device
    if args.dataset == 'online':
        dataset = CarlaOnlineEmbeddingDataset(embeddings_path=args.embeddings, json_path=args.metadata, sequence=args.use_sequence)
    else:
        dataset = CarlaEmbeddingDataset(embeddings_path=args.embeddings, json_path=args.metadata, sequence=args.use_sequence)

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=PadSequence(), drop_last=True)

    if args.use_sequence:
        temp_model = SequenceRNNEncoder(
            num_layers=args.num_layers, 
            hidden_size=args.hidden_size,
            action__chn=args.action_channels,
            speed_chn=args.speed_channels,
            bidirectional=args.bidirectional,
            )
    elif args.rnn_model == "vanilla":
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
        raise ValueError('Model not implemented')

    temp_model.load_state_dict(torch.load(args.temp_weights))
    temp_model.to(args.device)
    temp_model.eval()

    # create model
    vis_model = ADEncoder(backbone=args.backbone, use_timm=args.force_timm)
    vis_model.load_state_dict(torch.load(args.vis_weights))
    vis_model.to(args.device)
    vis_model.eval()

    h = temp_model.init_hidden(args.batch_size, device=device)

    sum_h = torch.zeros((4, 1024), device=args.device)
    sum_emb = torch.zeros((512, 4, 4), device=args.device)
    q = 0

    for i, (embeddings, embeddings_length, actions, speeds, embeddings_label) in enumerate(loader):
        embeddings, embeddings_label, actions, speeds = embeddings.to(args.device), embeddings_label.to(
            args.device), actions.to(
            args.device), speeds.to(args.device)
        pred_embedding, h = temp_model(embeddings, actions, speeds, hidden=h)

        output_expected = vis_model.decode(embeddings_label[:, -1, :, :, :])
        segmentation = output_expected['segmentation'].argmax(dim=1)

        output_predicted = vis_model.decode(pred_embedding[:, -1, :, :, :])
        pred_segmentation = output_predicted['segmentation'].argmax(dim=1)

        sum_h += h[0].sum(dim=1)
        sum_emb += embeddings[:, -1, :, :, :].sum(dim=0)
        q += actions.shape[0]

        if save_imgs:
            vmin = pred_segmentation.min()
            vmax = pred_segmentation.max()

            for j in range(pred_segmentation.shape[0]):
                pred_seg = pred_segmentation[j].detach().cpu().numpy()
                seg = segmentation[j].detach().cpu().numpy()
                save_image(pred_seg, '{}/{}-{}_pred.png'.format(args.img_folder, i, j), vmin=vmin, vmax=vmax)
                save_image(seg, '{}/{}-{}_real.png'.format(args.img_folder, i, j), vmin=vmin, vmax=vmax)

        # if i >= args.nb_images // args.batch_size + 1:
        #     break
    



    sigma_h = torch.zeros((4, 1024), device=args.device)
    sigma_emb = torch.zeros((512, 4, 4), device=args.device)
    for i, (embeddings, embeddings_length, actions, speeds, embeddings_label) in enumerate(loader):
        embeddings, embeddings_label, actions, speeds = embeddings.to(args.device), embeddings_label.to(
            args.device), actions.to(
            args.device), speeds.to(args.device)
        pred_embedding, h = temp_model(embeddings, actions, speeds, hidden=h)

        output_expected = vis_model.decode(embeddings_label[:, -1, :, :, :])
        segmentation = output_expected['segmentation'].argmax(dim=1)

        output_predicted = vis_model.decode(pred_embedding[:, -1, :, :, :])
        pred_segmentation = output_predicted['segmentation'].argmax(dim=1)


        sigma_h += (h[0].sum(dim=1) - sum_h/q)**2/q
        sigma_emb += (embeddings[:, -1, :, :, :].sum(dim=0) - sum_emb/q)**2/q

        if save_imgs:
            vmin = pred_segmentation.min()
            vmax = pred_segmentation.max()

            for j in range(pred_segmentation.shape[0]):
                pred_seg = pred_segmentation[j].detach().cpu().numpy()
                seg = segmentation[j].detach().cpu().numpy()
                save_image(pred_seg, '{}/{}-{}_pred.png'.format(args.img_folder, i, j), vmin=vmin, vmax=vmax)
                save_image(seg, '{}/{}-{}_real.png'.format(args.img_folder, i, j), vmin=vmin, vmax=vmax)

        # if i >= args.nb_images // args.batch_size + 1:
        #     break


    if plot_emb:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2)
        
        _ = ax[0, 0].hist((sum_h/q).view((-1)).detach().cpu().numpy(), bins='auto')
        ax[0, 0].set_title('Media h')
        _ = ax[0, 1].hist((sum_emb/q).view((-1)).cpu().numpy(), bins='auto')
        ax[0, 1].set_title('Media Embeddings')

        _ = ax[1, 0].hist((sigma_h).view((-1)).detach().cpu().numpy(), bins='auto')
        ax[1, 0].set_title('Var h')
        _ = ax[1, 1].hist((sigma_emb).view((-1)).cpu().numpy(), bins='auto')
        ax[1, 1].set_title('Var Embeddings')

        plt.show()