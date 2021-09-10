import json
from os import times
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse

PREFIX = 'clean_'

def get_episode_file(episode, metadata):
        date = metadata.name.split('.')[0]
        path = f"{metadata.parent}/{date}_{episode}.npz"
        p = Path(path)
        if p.is_file:
            return p
        print(f"Warning: data file not found {path}")

def episode_condition_delete(timestamp):
    if timestamp['collision']:
        # print(timestamp)
        return True
    return False

def timestamp_condition_detele(timestamp):
    return False

def clean_batch(metadata, metadata_path):
    for ep_key, episode_metadata in metadata.items():
        data_path = get_episode_file(ep_key, metadata_path)
        # episode_data = np.load(data_path)
        for t_key, timestamp in episode_metadata.items():
            if timestamp_condition_detele(timestamp):
                # Delete timestamp from metadata
                del metadata[ep_key][t_key]
                # Delete timestamp from affordances file
                # Not implemented yet

            if episode_condition_delete(timestamp):
                # delete episode from metadata
                del metadata[ep_key]
                # Delete episode affordances file
                # Not implemented yet
                print(f"Collision at Episode: {ep_key} Timestamp: {t_key}")
                break

    return metadata
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Settings for the data cleaning",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to data folder')

    args = parser.parse_args()

    p = Path(args.data)
    assert(p.is_dir())
    metadata_files = sorted(p.glob('**/**/*.json'))

    for path in tqdm(metadata_files):
        with open(path) as f:
            metadata = json.load(f)
            clean_metadata = clean_batch(metadata, path)
        
        with open(path, 'w') as f:
            json.dump(clean_metadata, f)