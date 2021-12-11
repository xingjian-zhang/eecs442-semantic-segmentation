import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self._double_conv(x)


class DownSampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._down_sample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvUnit(in_channels, out_channels)
        )

    def forward(self, x):
        return self._down_sample(x)


class UpSampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self._up_sample = nn.Sequential()
        if bilinear:
            self._up_sample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            self._up_sample = nn.ConvTranspose2d(
                in_channels, in_channels//2, kernel_size=2, stride=2)

        self._conv = DoubleConvUnit(in_channels, out_channels)

    def forward(self, x_prev, x_skip):
        """
        x_prev: (B, C, W, H)
        x_skip: (B, C, W, H)
        """
        x_prev = self._up_sample(x_prev)
        assert x_prev.shape == x_skip.shape, "Shape of x_prev and x_skip are different."
        x = torch.cat([x_prev, x_skip], dim=1)
        return self._conv(x)


class UNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_layers: int,
                 num_first_layer_features: int = 64,
                 bilinear: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.feature_list = [3, num_first_layer_features]
        self.layers = nn.ModuleDict()
        
        # Down sampling
        down_sample_layers = nn.ModuleList()
        down_sample_layers.append(DoubleConvUnit(3, num_first_layer_features))
        num_features = num_first_layer_features
        for _ in range(self.num_layers - 1):
            down_sample_layers.append(DownSampleUnit(num_features, num_features * 2))
            num_features *= 2
            self.feature_list.append(num_features)
        self.layers.add_module("downsample", down_sample_layers)

        # Up sampling
        up_sample_layers = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            up_sample_layers.append(UpSampleUnit(num_features, num_features // 2))
            num_features //= 2
            self.feature_list.append(num_features)
        self.layers.add_module("upsample", up_sample_layers)

        # Classification head
        self.layers.add_module("class", nn.Conv2d(num_features, num_classes, kernel_size=1))

    def forward(self, x):
        """
        x: (B, 3, W, H)
        """
        # Down sampling
        history = [x]
        for layer in self.layers["downsample"]:
            history.append(layer(history[-1]))
        history = history[1:]

        # Up sampling
        x = history[-1]
        for layer, x_skip in zip(self.layers["upsample"], history[::-1][1:]):
            x = layer(x, x_skip)
        
        # Classification
        return self.layers["class"](x)