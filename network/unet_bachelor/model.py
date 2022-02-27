import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

from network.unet_bachelor.layers import (
    DoubleConv,
    UpConv
)


class UNET_BACHELOR(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET_BACHELOR, self).__init__()

        features = [in_channels, 64, 128, 256, 512]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.unpool_correctances = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(features[2], out_channels, kernel_size=(1, 1))

        for i in range(len(features) - 2):
            self.down_convs.append(DoubleConv(features[i], features[i + 1]))

        self.bottleneck = DoubleConv(features[-2], features[-1])

        features = features[::-1]

        for i in range(len(features) - 2):
            if i != (len(features) - 3):
                self.up_convs.append(DoubleConv(features[i], features[i + 1]))
            else:
                self.up_convs.append(DoubleConv(features[i], features[i]))

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

            skip_connections[i] = tf.center_crop(skip_connections[i], output_size=x.shape[2:])
            go = torch.cat((x, skip_connections[i]), dim=1)
            x = self.up_convs[i](go)

        return self.final(x)


if __name__ == '__main__':
    unet = UNET_BACHELOR(in_channels=4, out_channels=1)

    y = torch.randn(1, 4, 1000, 1000)

    conv = nn.Conv2d(4, 4, kernel_size=(11, 11), stride=(1, 1), padding=5, padding_mode='reflect', bias=False)
    outnow = conv(y)

    print(outnow.shape)
