import argparse
import random
from pathlib import Path
from tqdm import tqdm
import json

def write_txt(array, path):
    with open(path, "w") as textfile:
        for element in array:
            textfile.write(element + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for the data capture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to data folder')
    parser.add_argument('--proportion', default=0.2, type=float, help='Proportion of the test set')

    args = parser.parse_args()
    data_folder = args.data
    p = Path(data_folder)
    assert(p.is_dir())

    metadata_files = sorted(p.glob('**/**/*.json'))
    episodes = []
    for path in tqdm(metadata_files, "Reading dataset..."):
        with open(path) as f:
            metadata = json.load(f)
            episodes += list(metadata.keys())

    split_idx = int(len(episodes) * args.proportion) - 1

    random.shuffle(episodes)

    train = episodes[split_idx:]
    test = episodes[:split_idx]

    write_txt(train, Path(data_folder).joinpath("train_keys.txt"))
    write_txt(test, Path(data_folder).joinpath("val_keys.txt"))

    print(f"Total train episodes: {len(train)}")
    print(f"Total validation episodes: {len(test)}")