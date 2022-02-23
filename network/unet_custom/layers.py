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

        self.laplace = torch.Tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

        self.laplace_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        residual = x
        x = self.double_conv(x)

        with torch.no_grad():
            self.laplace_conv.weight = nn.Parameter(self.laplace)

        residual = self.laplace_conv(residual)
        residual = self.batchnorm(residual)
        residual = self.relu(residual)

        return x, residual


class DoubleConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvUp, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        skip_connection = tf.resize(skip_connection, size=x.shape[2:])

        x = torch.concat((x, skip_connection), dim=1)

        return self.double_conv(x)


class DoubleConvBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBottleNeck, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


if __name__ == '__main__':
    a = torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    b = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    a = torch.add(a, b)

    print(a)
