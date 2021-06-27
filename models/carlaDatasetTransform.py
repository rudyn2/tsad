from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import torch
from pathlib import Path
import random
from torchvision.transforms import transforms

__CLASS_MAPPING__ = {
        0: 5,  # None
        1: 5,  # Buildings
        2: 5,  # Fences
        3: 5,  # Other
        4: 6,  # Pedestrians
        5: 5,  # Poles
        6: 2,  # RoadLines
        7: 3,  # Roads
        8: 4,  # Sidewalks
        9: 5,  # Vegetation
        10: 0,  # Vehicles
        11: 5,  # Walls
        12: 5,  # TrafficSigns
        13: 5,  # Sky
        14: 5,  # Ground
        15: 5,  # Bridge
        16: 5,  # RailTrack
        17: 5,  # GuardRail
        18: 1,  # Traffic Light
        19: 5,  # Static
        20: 5,  # Dynamic
        21: 5,  # Water
        22: 5  # Terrain
    }

class CarlaDatasetTransform(Dataset):

    # moving obstacles (0),  traffic lights (1),  road markers(2),  road (3),  sidewalk (4) and background (5).
    

    def __init__(self, path: str, prob: float):
        self.path = path
        self.run_timestamp_mapping = {}
        self.timestamps = None
        self.metadata = None
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.depth_transform = None

        self.composition = ActorComposition(__CLASS_MAPPING__[10], __CLASS_MAPPING__[4], __CLASS_MAPPING__[18], path=path, prob=prob)

        self.hdf5_path = self.read_timestamps()
        self.json_path = self.read_metadata()
        

    def read_timestamps(self):
        hdf5_files = list(Path(self.path).glob('**/*.hdf5'))
        if len(hdf5_files) < 1:
            raise RuntimeError("We couldn't find hdf5 files at provided path")

        # assume that there is just one hdf5 file at provided path
        hdf5_path = hdf5_files[0]
        with h5py.File(hdf5_path, "r") as f:
            for run in f.keys():
                for timestamp in f[run].keys():
                    self.run_timestamp_mapping[timestamp] = run
                    semantic = np.array(f[run][timestamp]['semantic'])
                    self.composition.detect(semantic, run, timestamp)
        self.timestamps = list(self.run_timestamp_mapping.keys())
        # self.timestamps = self.timestamps[:128]     # REMOVE THIS
        return hdf5_path

    def read_metadata(self):
        json_files = list(Path(self.path).glob('**/*.json'))
        if len(json_files) < 1:
            raise RuntimeError("We couldn't find json files at provided path")

        # assume that there is just one hdf5 file at provided path
        json_path = json_files[0]
        with open(json_path, "r") as f:
            self.metadata = json.load(f)
        return json_path

    def get_metadata(self):
        return self.metadata

    def get_timestamps(self):
        return self.timestamps

    def get_random_full_episode(self):
        # select random episode
        run_id = random.choice(list(self.metadata.keys()))
        timestamps = self.metadata[run_id].keys()
        idxs = [self.timestamps.index(t) for t in timestamps]
        camera, segmentation, traffic_light, vehicle_affordances, pedestrian = [], [], [], [], []
        for idx in sorted(idxs):
            x, s, tl, v_aff, pds = self[idx]
            camera.append(x)
            segmentation.append(s)
            traffic_light.append(tl)
            vehicle_affordances.append(v_aff)
            pedestrian.append(pds)
        return camera, segmentation, traffic_light, vehicle_affordances, pedestrian

    def __getitem__(self, item):
        element = self.timestamps[item]
        run_id = self.run_timestamp_mapping[element]
        with h5py.File(self.hdf5_path, "r") as f:
            images = f[run_id][element]
            rgb, semantic, depth = np.array(images['rgb']), np.array(images['semantic']), np.array(images['depth'])
        rgb, semantic, depth, proc = self.composition(rgb, semantic, depth)

        data = self.metadata[run_id][element]
        # x = np.concatenate((rgb, depth[:, :, np.newaxis]), axis=2)
        # x = np.transpose(x, axes=[2, 0, 1])
        if self.transform:
            rgb_transformed = self.transform(rgb)
        else:
            rgb_transformed = torch.from_numpy(rgb).float()
        if self.depth_transform:
            depth_transformed = self.depth_transform(depth)
        else:
            depth_transformed = torch.tensor(depth).unsqueeze(0) / 1000

        # stack depth
        x = torch.cat([depth_transformed.float(), rgb_transformed]).float()

        # get ground truth
        s = semantic[np.newaxis, :, :]
        s = torch.tensor(s, dtype=torch.int8)

        tl = torch.tensor([1, 0] if data['tl_state'] == 'Green' else [0, 1], dtype=torch.float16)
        v_aff = torch.tensor([data['lane_distance'], data['lane_orientation']]).float()
        sum_pds = (s == 6).sum()
        pds = torch.tensor([0, 1] if sum_pds > 100 else [1, 0]).float()
        proc = torch.tensor([proc], dtype=torch.uint8)

        return x, s, tl, v_aff, pds, proc

    

    def __len__(self):
        return len(self.timestamps)




class ActorComposition(object):
    def __init__(self, *ids, path='../dataset', prob=0):
        self._prob = prob
        self._ids = ids
        self.hdf5_path = list(Path(path).glob('**/*.hdf5'))[0]
        self._det_thresh = 20
        self._tr_thresh = 1

        self._actors = {}
        for id in ids:
            self._actors[id] = []
    
    def _map_classes(self, semantic):
        mapped_semantic = np.empty(semantic.shape)
        for k, v in __CLASS_MAPPING__.items():
            mapped_semantic[semantic == k] = v
        return mapped_semantic
    
    def detect(self, semantic, run, timestamp):
        semantic = self._map_classes(semantic)
        for id in self._ids:
            if np.sum(semantic == id) > self._det_thresh:
                self._actors[id].append({
                    'run': run,
                    'timestamp': timestamp
                })
    
    def transform(self, rgb1, semantic1, depth1, rgb2, semantic2, depth2, id):
        semantic_mask = (semantic2 == id)
        depth_mask = (depth2 < depth1)
        mask = np.logical_and(semantic_mask, depth_mask)
        rgb1[mask] = rgb2[mask]
        depth1[mask] = depth2[mask]
        semantic1[mask] = semantic2[mask]
        return rgb1, semantic1, depth1
    
    def __call__(self, rgb1, semantic1, depth1):
        semantic1 = self._map_classes(semantic1)
        processed = False

        for id in self._ids:
            #if len(self._actors[id]) == 0:
            #    print(f"No examples for {id}")
            if np.sum(semantic1 == id)/semantic1.size <= self._tr_thresh and np.random.rand() < self._prob and len(self._actors[id]) > 0:
                processed = True
                random_idx = np.random.choice(range(len(self._actors[id])))
                random_sample = self._actors[id][random_idx]
                run_id = random_sample['run']
                element = random_sample['timestamp']
                with h5py.File(self.hdf5_path, "r") as f:
                    sample2 = f[run_id][element]
                    rgb2, semantic2, depth2 = np.array(sample2['rgb']), self._map_classes(np.array(sample2['semantic'])), np.array(sample2['depth'])
                    rgb1, semantic1, depth1 = self.transform(rgb1, semantic1, depth1, rgb2, semantic2, depth2, id)

        return rgb1, semantic1, depth1, processed




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from carlaDatasetSimple import CarlaDatasetSimple
    import time

    d1 = CarlaDatasetSimple('../dataset')
    d2 = CarlaDatasetTransform('../dataset', prob=1.)

    id = 1000

    actors = d2.composition._actors
    print(f"Number of images with pedestrians: {len(actors[6])}\tNumber of images with vehicles: {len(actors[0])}\tNumber of images with TL: {len(actors[1])}")

    tic = time.time()
    x1, s1, tl1, v_aff1, pds1 = d1[id]
    t1 = time.time() - tic
    tic = time.time()
    x2, s2, tl2, v_aff2, pds2, proc = d2[id]
    t2 = time.time() - tic
    print(f"Time SimpleDataset: {t1}\t Time TransformDataset: {t2}")
    print(f"Processed: {proc[0]}")

    image = np.transpose(x1.cpu().numpy(), axes=(1, 2, 0))[:, :, -3:]
    print(f"\nNumber before ped pixels: {np.sum(s1.cpu().numpy() == 6)}")
    print(f"Number before tl pixels: {np.sum(s1.cpu().numpy() == 1)}")
    print(f"Number before ve pixels: {np.sum(s1.cpu().numpy() == 0)}")
    depth = x1.cpu().numpy()[0]

    fig, axs = plt.subplots(2, 3)
    axs[0,0].set_title('RGB original')
    axs[0,0].imshow(image, vmin=0, vmax=1)

    axs[0,1].set_title('Semantic original')
    axs[0,1].imshow(s1.cpu().numpy()[0], vmin=0, vmax=6)
    #print(s1.cpu().numpy()[0][170:200, 190:200])
    #print(s1.cpu().numpy()[0][170:200, 190:220])

    axs[0,2].set_title('Depth original')
    axs[0,2].imshow(np.log(depth), cmap='gray')



    image = np.transpose(x2.cpu().numpy(), axes=(1, 2, 0))[:, :, -3:]
    print(f"\nNumber after ped pixels: {np.sum(s2.cpu().numpy() == 6)}")
    print(f"Number after tl pixels: {np.sum(s2.cpu().numpy() == 1)}")
    print(f"Number after ve pixels: {np.sum(s2.cpu().numpy() == 0)}")
    depth = x2.cpu().numpy()[0]
    axs[1,0].set_title('RGB modified')
    axs[1,0].imshow(image, vmin=0, vmax=1)

    axs[1,1].set_title('Semantic modified')
    axs[1,1].imshow(s2.cpu().numpy()[0], vmin=0, vmax=6)

    axs[1,2].set_title('Depth modified')
    axs[1,2].imshow(np.log(depth), cmap='gray')
    plt.show()
    #print(d.get_random_full_episode())
