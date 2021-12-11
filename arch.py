import torch
from torchsummary import summary

from config import IMAGE_HEIGHT, IMAGE_WIDTH
from model import UNET

layers = [1, 2, 3, 4, 5]
features = [64, 128, 256, 512, 1024]
for l in layers:
    print("LAYERS = ", l)
    u = UNET(3, 1, features=features[:l]).cuda()
    summary(u, (3, 256, 256))
    print("\n")
