import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        return self.up(x)


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


class UnfanningAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnfanningAttention, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.breakdown = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.ConvTranspose2d(1, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode='zeros')

    def forward(self, small, big):
        save = torch.add(small, big)

        skip_con = self.relu(save)

        skip_con = self.breakdown(skip_con)
        skip_con = self.sigmoid(skip_con)
        skip_con = self.upsample(skip_con)

        skip_con = torch.multiply(skip_con, save)

        return self.batchnorm(skip_con)
