import torch
import torch.nn as nn

import torchvision.transforms.functional as tf

from layers_pl import (
    BasicBlock,
    GFAM
)


def normalize_tensor(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


class PLNET(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(PLNET, self).__init__()

        self.in_channels = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.gfam1 = GFAM(256, 512)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        gfam1_res = self.gfam1(x3, x4, None)

        return x


if __name__ == '__main__':
    plnet = PLNET(BasicBlock, [3, 4, 6, 3])

    a = torch.randn(1, 4, 512, 512)
    out = plnet(a)
