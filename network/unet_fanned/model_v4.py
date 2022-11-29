import torch
import torch.nn as nn

import torchvision.transforms.functional as tf

from unet_fanned.layers import (
    DoubleConv_Small,
    DoubleConv_Big,
    UpConv
)


class UNET_FANNED(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNET_FANNED, self).__init__()

        features = [in_channels, 64, 128, 256, 512, 1024]

        self.down_convs_small = nn.ModuleList()
        self.down_convs_big = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(features[1], out_channels, kernel_size=(1, 1))

        for i in range(len(features) - 2):
            self.down_convs_small.append(DoubleConv_Small(features[i], features[i + 1]))
            self.down_convs_big.append(DoubleConv_Big(features[i], features[i + 1]))

        self.bottleneck_small = DoubleConv_Small(features[-2], features[-1])
        self.bottleneck_big = DoubleConv_Big(features[-2], features[-1])

        features = features[::-1]

        for i in range(len(features) - 2):
            self.up_convs.append(DoubleConv_Small(features[i], features[i + 1]))
            self.up_trans.append(UpConv(features[i], features[i + 1]))

    def forward(self, x, y):
        skip_connections_small = []
        skip_connections_big = []

        for i in range(len(self.down_convs_small)):
            x = self.down_convs_small[i](x)
            y = self.down_convs_big[i](y)

            skip_connections_small.append(x)
            skip_connections_big.append(y)

            x = self.pool(x)
            y = self.pool(y)

        x = self.bottleneck_small(x)
        y = self.bottleneck_big(y)

        bottleneck_unfanned = torch.add(x, y)

        skip_connections_small = skip_connections_small[::-1]
        skip_connections_big = skip_connections_big[::-1]

        skip_connections_unfanned = []

        for i in range(len(skip_connections_small)):
            skip_connections_unfanned.append(torch.add(skip_connections_small[i], skip_connections_big[i]))

        for i in range(len(self.up_convs)):
            bottleneck_unfanned = self.up_trans[i](bottleneck_unfanned, skip_connections_unfanned[i])

            bottleneck_unfanned = torch.cat((bottleneck_unfanned, skip_connections_unfanned[i]), dim=1)
            bottleneck_unfanned = self.up_convs[i](bottleneck_unfanned)

        return self.final(bottleneck_unfanned)


if __name__ == '__main__':
    unet = UNET_FANNED(in_channels=4, out_channels=1)

    a = torch.randn(1, 4, 512, 512)
    out = unet(a, a)

    print(out.shape)
