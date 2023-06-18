import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class GFAM(nn.Module):
    def __init__(self, block1_channels, feat2_channels):
        super(GFAM, self).__init__()

        self.block1_conv1 = nn.Conv2d(in_channels=block1_channels, out_channels=block1_channels, kernel_size=(1, 1))

        self.feat2_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.feat2_conv1 = nn.Conv2d(in_channels=feat2_channels, out_channels=feat2_channels // 2, kernel_size=(1, 1))

        self.middel_conv = nn.Conv2d(in_channels=block1_channels, out_channels=block1_channels, kernel_size=(1, 1))

        self.result_conv1 = nn.Conv2d(in_channels=feat2_channels, out_channels=feat2_channels, kernel_size=(3, 3),
                                      padding=1)
        self.result_conv2 = nn.Conv2d(in_channels=feat2_channels, out_channels=feat2_channels, kernel_size=(3, 3),
                                      padding=1)

    def forward(self, block1, feat2, resmap):
        block1 = self.block1_conv1(block1)

        feat2 = self.feat2_upsampling(feat2)
        feat2 = self.feat2_conv1(feat2)

        addition = torch.add(block1, feat2)

        if resmap is not None:
            oners = torch.ones(resmap.shape)
            nofr = torch.sub(oners, resmap)

            concat_middle = torch.cat((addition, nofr), axis=1)
            concat_middle = self.middel_conv(concat_middle)
            concat_middle = torch.sigmoid(concat_middle)
        else:
            concat_middle = addition

        upper_result = torch.mul(block1, concat_middle)

        before_conv = torch.cat((upper_result, feat2), axis=1)

        result = self.result_conv1(before_conv)
        result = self.result_conv2(result)

        return result
