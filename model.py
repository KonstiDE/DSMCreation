import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as tf

from layers import (
    DoubleConv,
    UpConv
)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET, self).__init__()

        features = [in_channels, 64, 128, 256, 512, 1024]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(features[2], out_channels, kernel_size=(1, 1))
        self.dropout = nn.Dropout2d(p=0.5)

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

            skip_connection = skip_connections[i]

            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:])

            go = torch.cat((x, skip_connection), dim=1)
            x = self.up_convs[i](go)

        return self.final(x)


if __name__ == '__main__':
    unet = UNET(in_channels=3, out_channels=1).cuda()

    x = torch.randn(1, 3, 500, 500).cuda()

    out = unet(x)

    print(out.shape)
