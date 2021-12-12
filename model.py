import time
import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms.functional as TF
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import pandas as pd
import matplotlib.pyplot as plt

from config import LEARNING_RATE, WEIGHT_DECAY


class DoubleConv(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        kaiming_uniform_(self.conv[0].weight, nonlinearity='relu')
        kaiming_uniform_(self.conv[3].weight, nonlinearity='relu')

    def forward(self, x):
        return self.conv(x)


class UNET(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_time = 0
        self.layers = len(features)
        self.logfile = "unet_l{}.log".format(self.layers)

        # Down part of Unet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of Unet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        data = data.float()
        targets = targets.float().unsqueeze(1)
        predictions = self.forward(data)
        loss = self.criterion(predictions, targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 'min', factor=.75, patience=5, verbose=5, eps=1e-6)
        return {"optimizer": optimizer}

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        data = data.float()
        targets = targets.float().unsqueeze(1)
        predictions = self.forward(data)
        os.makedirs("predictions/", exist_ok=True)
        self.visualize([data[0], targets[0], predictions[0]],
                       file_path="predictions/l{}_epoch{}_{}.jpeg".format(self.layers, self.current_epoch, batch_idx))
        loss = self.criterion(predictions, targets)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions >= .5).float()
        num_correct = (predictions == targets).sum()
        num_pixels = torch.numel(predictions)
        dice_score = (2 * (predictions * targets).sum()) / \
            ((predictions + targets).sum() + 1e-8)
        accuracy = num_correct / num_pixels
        metrics = {'val_loss': loss, "val_accuracy": accuracy,
                   "dice_score": dice_score}
        return metrics

    def validation_epoch_end(self, outputs):
        df = pd.DataFrame(outputs)
        time.sleep(0.5)  # Avoid messing up progress bar
        with open(self.logfile, "a") as fp:
            fp.write("\n val_loss: {}\n val_accuracy: {}\n dice_score: {}\n".format(
                df["val_loss"].mean(),
                df["val_accuracy"].mean(),
                df["dice_score"].mean()
            ))

    @staticmethod
    def visualize(display_list, title_list=["Input", "True Mask", "Pred Mask"], file_path=None):
        plt.figure(figsize=(10, 5))
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title_list[i])
            plt.imshow(display_list[i].cpu().numpy().transpose(1, 2, 0))
            plt.axis("off")
        plt.savefig(file_path, dpi=300)

    def on_train_epoch_start(self) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(self, *args) -> None:
        self.end_time = time.time()
        self.train_time += self.end_time - self.start_time
