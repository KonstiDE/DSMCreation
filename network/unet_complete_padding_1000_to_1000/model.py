import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

from network.unet_complete_padding_1000_to_1000.layers import (
    DoubleConvDown,
    DoubleConvUp,
    DoubleConvBottleNeck,
    UpSample
)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET, self).__init__()

        features = [in_channels, 64, 128, 256, 512]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(features[1], out_channels, kernel_size=(1, 1))

        for i in range(len(features) - 2):
            self.down_convs.append(DoubleConvDown(features[i], features[i + 1]))

        self.bottleneck = DoubleConvBottleNeck(features[-2], features[-1])

        features = features[::-1]

        for i in range(0, len(features) - 2):
            self.up_convs.append(DoubleConvUp(features[i + 1]))
            self.up_trans.append(UpSample(features[i], features[i + 1]))

    def forward(self, x):
        skip_connections = []

        for down_conv in self.down_convs:
            print(x.shape)
            x, residual = down_conv(x)
            skip_connections.append(residual)
            x = self.pool(x)

        x = self.bottleneck(x)
        print(x.shape)

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.up_convs)):
            x = self.up_trans[i](x)
            x = self.up_convs[i](x, skip_connections[i])

        return self.final(x)


if __name__ == '__main__':
    unet = UNET(in_channels=4, out_channels=1)

    y = torch.randn(1, 4, 1000, 1000)

    out = unet(y)

    print(out.shape)
