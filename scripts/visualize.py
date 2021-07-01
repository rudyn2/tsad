import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np


def depth2grayscale(arr):
    arr[arr == 1000] = 0
    normalized_depth = cv2.normalize(arr, arr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    normalized_depth = np.stack((normalized_depth,) * 3, axis=-1)  # Grayscale into 3 channels
    return normalized_depth


def labels_to_cityscapes_palette(arr):
    classes = {
        0: [0, 0, 0],  # None
        1: [70, 70, 70],  # Buildings
        2: [100, 40, 40],  # Fences
        3: [55, 90, 80],  # Other
        4: [220, 20, 60],  # Pedestrians
        5: [153, 153, 153],  # Poles
        6: [157, 234, 50],  # RoadLines
        7: [128, 64, 128],  # Roads
        8: [244, 35, 232],  # Sidewalks
        9: [107, 142, 35],  # Vegetation
        10: [0, 0, 142],  # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0],  # TrafficSigns
        13: [70, 130, 180],  # Sky
        14: [81, 0, 81],  # Ground
        15: [150, 100, 100],  # Bridge
        16: [230, 150, 140],  # RailTrack
        17: [180, 165, 180],  # GuardRail
        18: [250, 170, 30],  # Traffic Light
        19: [110, 190, 160],  # Static
        20: [170, 120, 50],  # Dynamic
        21: [45, 60, 150],  # Water
        22: [145, 170, 100]  # Terrain
    }

    result = np.zeros((arr.shape[0], arr.shape[1], 3))
    for key, value in classes.items():
        result[np.where(arr == key)] = value
    result /= 255.0
    return result


def main(file_path: str, metadata: str):
    f = h5py.File(file_path, "r")
    total_run = sum([k.startswith("run") for k in f.keys()])
    total_frames = 0
    for k in f.keys():
        timestamps = list(f[k].keys())
        total_frames += len(timestamps)
    print(f"The file: {file_path} contains {total_run} ego runs")
    print(f"The file contains {total_frames} frames\n")

    ego_info = None
    if metadata is not None and os.path.exists(metadata):
        with open(metadata, "r+") as file:
            ego_info = json.load(file)

    for i, run_id in enumerate(f.keys()):
        print(f"[{i}]: {run_id}")

    print("")
    opt = input("Select an option: ")
    opt = list(f.keys())[int(opt)]
    ego_run_imgs = f[opt]
    if ego_info:
        ego_run_info = ego_info[opt]
    else:
        ego_run_info = None

    timestamps = list(ego_run_imgs.keys())
    print(f"{len(timestamps)} frames for selected record")

    for i, timestamp in enumerate(timestamps):

        rgb_image = np.array(ego_run_imgs[timestamp]['rgb'])
        depth_map = np.array(ego_run_imgs[timestamp]['depth'])
        semantic_map = np.array(ego_run_imgs[timestamp]['semantic'])
        cv2.imshow('rgb', rgb_image[64:, :, :])
        cv2.imshow('depth', depth2grayscale(depth_map))
        cv2.imshow('semantic', labels_to_cityscapes_palette(semantic_map))

        if ego_run_info:
            sys.stdout.write("\r")
            if ego_run_info[timestamp]["at_tl"]:
                s = f'Frame {i + 1} | {ego_run_info[timestamp]["command"]} | ' \
                    f'Light state: {ego_run_info[timestamp]["tl_state"]} | Distance to TL: {ego_run_info[timestamp]["tl_distance"]:.3f} ' \
                    f'| Speed: {ego_run_info[timestamp]["speed"]:.2f} | Distance to center: {ego_run_info[timestamp]["lane_distance"]:.3f} ' \
                    f'| Orientation: {ego_run_info[timestamp]["lane_orientation"]:.3f}'
            else:
                s = f'Frame {i + 1} | {ego_run_info[timestamp]["command"]} | ' \
                    f'Speed: {ego_run_info[timestamp]["speed"]:.2f} | Distance to center: {ego_run_info[timestamp]["lane_distance"]:.3f} ' \
                    f'| Orientation: {ego_run_info[timestamp]["lane_orientation"]:.3f}'
            sys.stdout.write(s)
            sys.stdout.flush()

        cv2.waitKey(50)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HDF5 Carla Dataset Visualization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, default=None, type=str, help='path to hdf5 file')
    parser.add_argument('-m', '--metadata', default=None, type=str, help='path to json file')
    args = parser.parse_args()
    main(args.input, args.metadata)
