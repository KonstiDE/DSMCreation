import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

        self.conv_gating = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.conv_skipcon = nn.Conv2d(int(in_channels / 2), out_channels, kernel_size=(1, 1), stride=(2, 2))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.breakdown = nn.Conv2d(int(in_channels / 2), 1, kernel_size=(1, 1))
        self.up = nn.ConvTranspose2d(1, out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, g, x):
        save_skipcon = x

        g = self.conv_gating(g)
        x = self.conv_skipcon(x)

        #combined = torch.add(x, g)

        #combined = self.relu(combined)
        #combined = self.breakdown(combined)
        #combined = self.sigmoid(combined)
        #combined = self.up(combined)
        #go = torch.multiply(self.up(self.sigmoid(self.breakdown(self.relu(torch.add(x, g))))), save_skipcon)
        #return self.batchnorm(go)

        return self.batchnorm(torch.multiply(self.up(self.sigmoid(self.breakdown(self.relu(torch.add(x, g))))), save_skipcon))


class DoubleConv_Small(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_Small, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)
        self.correctance = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, stride=(1, 1))

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        save = x
        x = self.double_conv(x)

        if self.in_channels != self.out_channels:
            save = self.correctance(save)

        x = torch.add(x, save)

        return self.relu(x)


class DoubleConv_Big(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_Big, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)
        self.correctance = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False, stride=(1, 1))

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(7, 7), bias=False, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 7), bias=False, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        save = x
        x = self.double_conv(x)

        if self.in_channels != self.out_channels:
            save = self.correctance(save)

        x = torch.add(x, save)

        return self.relu(x)
