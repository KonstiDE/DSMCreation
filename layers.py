import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import numpy as np


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        return self.up(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.correctance = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1)

        )

    def forward(self, x):
        save = x
        x = self.double_conv(x)
        save = self.correctance(save)

        torch.add(x, save)
        x = self.batchnorm(x)

        return self.relu(x)
