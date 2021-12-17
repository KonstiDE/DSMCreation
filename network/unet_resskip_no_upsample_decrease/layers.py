import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import numpy as np


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        return self.up(x)


class DoubleConvDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvDown, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        residual = x
        x = self.double_conv(x)

        residual = self.residual(residual)

        return x, residual


class DoubleConvUp(nn.Module):
    def __init__(self, channels):
        super(DoubleConvUp, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.double_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, residual):

        residual = self.residual(x)
        x = self.double_conv(x)

        torch.add(x, tf.center_crop(residual, output_size=x.shape[2:]))

        return residual


class DoubleConvBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBottleNeck, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
