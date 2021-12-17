import torch
import numpy as np
from torchsummary import summary

from config import IMAGE_HEIGHT, IMAGE_WIDTH
from model import UNET

scale = [1, 2, 4, 8, 16]
features = [64, 128, 256, 512]
features = np.array(features)
for l in scale:
    print("channels = ", features[0]//l)
    u = UNET(3, 1, features=features//l).cuda()
    summary(u, (3, 256, 256))
    print("\n")
