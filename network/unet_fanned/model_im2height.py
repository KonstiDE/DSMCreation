import torch
import torch.nn as nn
import torch.nn.functional as F


class Pool(nn.Module):
    def __init__(self, kernel_size=2, stride=2, **kwargs):
        super(Pool, self).__init__()

        self.pool_fn = nn.MaxPool2d(kernel_size, stride, **kwargs)

    def forward(self, x, *args, **kwargs):
        size = x.size()
        x, indices = self.pool_fn(x, **kwargs)

        return x, indices, size


class Unpool(nn.Module):
    def __init__(self, fn, kernel_size=2, stride=2, **kwargs):
        super(Unpool, self).__init__()

        self.pool_fn = nn.MaxUnpool2d(kernel_size, stride, **kwargs)

    def forward(self, x, indices, output_size, *args, **kwargs):
        return self.pool_fn(x, indices=indices, output_size=output_size, *args, **kwargs)


class Block(nn.Module):
    def __init__(self, fn, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()

        self.conv1 = fn(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_rest = fn(out_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        # following similar setup https://github.com/hysts/pytorch_resnet
        self.identity = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.identity.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.identity.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn(self.conv1(x)))
        y = F.relu(self.bn(self.conv_rest(y)))
        y = self.bn(self.conv_rest(y))
        identity = self.identity(x)
        y = F.relu(y + identity)

        return y


class IM2HEIGHT(nn.Module):
    def __init__(self):
        super(IM2HEIGHT, self).__init__()

        # Convolutions
        self.conv1 = Block(nn.Conv2d, 4, 64)
        self.conv2 = Block(nn.Conv2d, 64, 128)
        self.conv3 = Block(nn.Conv2d, 128, 256)
        self.conv4 = Block(nn.Conv2d, 256, 512)

        # Deconvolutions
        self.deconv1 = Block(nn.ConvTranspose2d, 512, 256)
        self.deconv2 = Block(nn.ConvTranspose2d, 256, 128)
        self.deconv3 = Block(nn.ConvTranspose2d, 128, 64)
        self.deconv4 = Block(nn.ConvTranspose2d, 128, 1)  # note this is residual merge

        self.pool = Pool(2, 2, return_indices=True)
        self.unpool = Unpool(2, 2)

    def forward(self, x):
        # Convolve
        x = self.conv1(x)
        # Residual skip connection
        x_conv_input = x.clone()
        x, indices1, size1 = self.pool(x)
        x, indices2, size2 = self.pool(self.conv2(x))
        x, indices3, size3 = self.pool(self.conv3(x))
        x, indices4, size4 = self.pool(self.conv4(x))

        # Deconvolve
        x = self.unpool(x, indices4, indices3.size())
        x = self.deconv1(x)
        x = self.unpool(x, indices3, indices2.size())
        x = self.deconv2(x)
        x = self.unpool(x, indices2, indices1.size())
        x = self.deconv3(x)
        x = self.unpool(x, indices1, x_conv_input.size())

        # Concatenate with residual skip connection
        x = torch.cat((x, x_conv_input), dim=1)
        x = self.deconv4(x)

        return x


if __name__ == '__main__':
    im2height = IM2HEIGHT()

    a = torch.randn(1, 4, 512, 512)
    out = im2height(a)

    print(out.shape)
