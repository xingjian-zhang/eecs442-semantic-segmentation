"""Boundary detection poc.

See https://www.kaggle.com/c/carvana-image-masking-challenge for details of task.
There are 3 additional packages required in this example:
    - scikit-learn
    - pandas
    - albumentations
"""
import argparse
import os
from zipfile import ZipFile
import warnings

import pytorch_lightning as pl
from orca_lite import init_orca_lite
from orca_lite.learn import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TEST_BATCH_SIZE
from dataset import CarvanaDataset
from model import UNET
from utils import get_df

parser = argparse.ArgumentParser()
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
parser.add_argument('--data_path', default='../../data/carvana-image-masking-challenge',
                    help='path to carvana-image-masking-challenge')
parser.add_argument('-j', '--jemalloc', action='store_false', default=True,
                    help='whether to use jemalloc')
parser.add_argument('-o', '--openmp', action='store_false', default=True,
                    help='whether to use openmp')
parser.add_argument('--num_processes', default=1, type=int,
                    help='number of processes using DDP')
args = parser.parse_args()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)  # Supress pytorch dataset UserWarning
    init_orca_lite(args.jemalloc, args.openmp)

    # Extract files
    targets = ["train", "train_masks", "train_masks.csv"]
    for target in targets:
        extract_path = os.path.join(args.data_path, target)
        if not os.path.exists(extract_path):
            print("Unzip training data {}.".format(extract_path))
            zip_path = os.path.join(args.data_path, target + ".zip")
            with ZipFile(zip_path, 'r') as zip_:
                zip_.extractall(path=args.data_path)
        else:
            print("Data {} already extracted.".format(extract_path))

    # Initialize datasets
    train_dir = os.path.join(args.data_path, "train/")
    mask_dir = os.path.join(args.data_path, "train_masks/")
    df, mask_df = get_df(train_dir, is_mask=False), get_df(mask_dir, is_mask=True)
    df["mask_path"] = mask_df["mask_path"]

    train_df, valid_df = train_test_split(df, random_state=2103, test_size=.2)

    train_dataset = CarvanaDataset(train_df, "train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_dataset = CarvanaDataset(valid_df, "test")
    valid_loader = DataLoader(valid_dataset, batch_size=TEST_BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print("# of train: {} # of val: {}".format(len(train_dataset), len(valid_dataset)))

    # Initialize model
    model = UNET(3, 1)

    # Train
    if args.ipex:
        print("NUM_PROCESSES PER CPU:", args.num_processes)
        trainer = Trainer(max_epochs=1, num_processes=args.num_processes)
    else:
        trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_loader, valid_loader)
    throughput = len(train_dataset) / model.train_time

    print("average throughput {}".format(throughput))


if __name__ == "__main__":
    main()
