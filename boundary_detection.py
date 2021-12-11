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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np

from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TEST_BATCH_SIZE
from dataset import CarvanaDataset
from model import UNET
from utils import get_df

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/',
                    help='path to carvana-image-masking-challenge')
parser.add_argument('--layers', default=4, type=int, help='number of layers in encoder')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--channel_scale', default=1, type=int, help='scale down')
args = parser.parse_args()


def main():

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
    features = [64, 128, 256, 512, 1024]
    features = np.array(features)
    features //= args.scale    
    model = UNET(3, 1, features=features[:args.layers])

    # Train
    trainer = pl.Trainer(max_epochs=5, devices=[args.gpu,], auto_select_gpus=True, accelerator='gpu')
    trainer.fit(model, train_loader, valid_loader)
    throughput = len(train_dataset) / model.train_time

    print("average throughput {}".format(throughput))


if __name__ == "__main__":
    main()
