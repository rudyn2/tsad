import sys

import cv2
import numpy as np
import torch

from models.ADEncoder import ADEncoder
from models.carlaDatasetSimple import CarlaDatasetSimple
from scripts.visualize import labels_to_cityscapes_palette

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../dataset', type=str, help='Path to dataset folder')
    parser.add_argument('--model', required=True, type=str, help='Path to model weights')
    parser.add_argument('--backbone', default='efficientnet-b5', type=str, help='Type of backbone')
    parser.add_argument('--device', default='cuda', type=str, help='Device to be used for evaluation')
    parser.add_argument('--debug', default='store_true', help='If true you can visualize results from one episode')
    args = parser.parse_args()

    print("Initializing...")
    dataset = CarlaDatasetSimple(args.data)
    model = ADEncoder(backbone=args.backbone, use_bn=True)
    model.to(args.device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    print("Showing predictions...\n")
    if args.debug:
        episode = dataset.get_random_full_episode()
        print(f"Length episode: {len(episode[0])}")
        for x, s, tl, v_aff, ped in zip(*episode):
            x = x.unsqueeze(0).to(args.device)
            pred = model(x)
            pred_segmentation = pred['segmentation'].squeeze().argmax(0).cpu().detach().numpy()
            pred_traffic_light_status = pred['traffic_light_status'].argmax().item()
            pred_vehicle_affordances = pred['vehicle_affordances'].cpu().detach().numpy()
            pred_pedestrian = pred['pedestrian'].cpu().detach().numpy()

            real_seg = s.squeeze().numpy()
            real_tl_status = 'Red' if tl.numpy().argmax() == 1 else 'Green'
            real_vh = v_aff.numpy()

            sys.stdout.write('\r')
            log = f"TL: {'Red' if pred_traffic_light_status == 1 else 'Green'} | {real_tl_status}, " \
                  f"Vehicle position: {pred_vehicle_affordances[0][0]:.3f} | {real_vh[0]:.3f}, " \
                  f"Vehicle orientation: {pred_vehicle_affordances[0][1]:.3f} |  {real_vh[1]:.3f}, " \
                  f"Pedestrian: {pred_pedestrian[0].argmax()}|{ped.argmax().item()}"
            sys.stdout.write(log)
            sys.stdout.flush()

            rgb_depth = np.transpose(x.squeeze().cpu().numpy(), axes=[1, 2, 0])
            rgb_input = rgb_depth[:, :, 1:]
            depth = rgb_depth[:, :, 0]
            cv2.imshow('RGB Input', rgb_input)
            # cv2.imshow('Depth', depth2grayscale(depth * 1000))
            cv2.imshow('Real segmentation', labels_to_cityscapes_palette(real_seg))
            cv2.imshow('Predicted segmentation', labels_to_cityscapes_palette(pred_segmentation))
            cv2.waitKey(100)
        cv2.destroyAllWindows()
    # evaluate metrics
