import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

from network.unet_bachelor.layers import (
    DoubleConv_Small,
    DoubleConv_Big,
    UpConv
)


class UNET_FANNED(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET_FANNED, self).__init__()

        features = [in_channels, 64, 128, 256, 512]

        self.down_convs_small = nn.ModuleList()
        self.down_convs_big = nn.ModuleList()
        self.unfanning = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.unpool_correctances = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(features[2], out_channels, kernel_size=(1, 1))
        self.bottleneck_unfanning = nn.Conv2d(features[-1] * 2, features[-1], kernel_size=(5, 5), bias=False, padding=2)

        for i in range(len(features) - 2):
            self.down_convs_small.append(DoubleConv_Small(features[i], features[i + 1]))
            self.down_convs_big.append(DoubleConv_Big(features[i], features[i + 1]))

        for i in range(len(features) - 2):
            self.unfanning.append(nn.Conv2d(features[i + 2], features[i + 1], kernel_size=(5, 5), bias=False, padding=2))

        self.bottleneck_small = DoubleConv_Small(features[-2], features[-1])
        self.bottleneck_big = DoubleConv_Big(features[-2], features[-1])

        features = features[::-1]

        for i in range(len(features) - 2):
            if i != (len(features) - 3):
                self.up_convs.append(DoubleConv_Small(features[i], features[i + 1]))
            else:
                self.up_convs.append(DoubleConv_Small(features[i], features[i]))

            self.up_trans.append(UpConv(features[i], features[i + 1]))

    def forward(self, x, y):
        skip_connections_small = []
        skip_connections_big = []

        z = None

        for i in range(len(self.down_convs_small)):
            x = self.down_convs_small[i](x)
            y = self.down_convs_big[i](y)

            skip_connections_small.append(x)
            skip_connections_big.append(y)

            x = self.pool(x)
            y = self.pool(y)

        x = self.bottleneck_small(x)
        y = self.bottleneck_big(y)

        bottleneck_unfanned = torch.cat((y, x), dim=1)
        bottleneck_unfanned = self.bottleneck_unfanning(bottleneck_unfanned)

        skip_connections_small = skip_connections_small[::-1]
        skip_connections_big = skip_connections_big[::-1]
        self.unfanning = self.unfanning[::-1]

        skip_connections_unfanned = []

        for i in range(len(skip_connections_small)):
            z_skip_connection = torch.concat((skip_connections_small[i], skip_connections_big[i]), dim=1)
            skip_connections_unfanned.append(self.unfanning[i](z_skip_connection))

        z = bottleneck_unfanned
        del bottleneck_unfanned

        for i in range(len(self.up_convs)):
            z = self.up_trans[i](z)

            go = torch.cat((z, skip_connections_unfanned[i]), dim=1)
            z = self.up_convs[i](go)

        return self.final(z)


if __name__ == '__main__':
    unet = UNET_FANNED(in_channels=4, out_channels=1)

    a = torch.randn(1, 4, 1000, 1000)
    out = unet(a, a)

    print(out.shape)
