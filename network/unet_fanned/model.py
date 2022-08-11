import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

from unet_fanned.layers import (
    DoubleConv,
    UpConv
)


class UNET_FANNED(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=None):
        super(UNET_FANNED, self).__init__()

        if features is None:
            features = [in_channels, 64, 128, 256, 512, 1024]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.unpool_correctances = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(features[1], out_channels, kernel_size=(1, 1))

        for i in range(len(features) - 2):
            self.down_convs.append(DoubleConv(features[i], features[i + 1]))

        self.bottleneck = DoubleConv(features[-2], features[-1])

        features = features[::-1]

        for i in range(len(features) - 2):
            self.up_convs.append(DoubleConv(features[i], features[i + 1]))
            self.up_trans.append(UpConv(features[i], features[i + 1]))

    def forward(self, x):
        skip_connections = []

        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.up_convs)):
            x = self.up_trans[i](x)
            go = torch.cat((x, skip_connections[i]), dim=1)
            x = self.up_convs[i](go)

        return self.final(x)
