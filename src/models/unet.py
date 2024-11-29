import torch.nn as nn
from .layers import DoubleConv, DownConv, UpConv

class Unet(nn.Module):
    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        self.first_conv = DoubleConv(channels_in, channels)
        self.down_conv1 = DownConv(channels, 2 * channels)
        self.down_conv2 = DownConv(2 * channels, 4 * channels)
        self.down_conv3 = DownConv(4 * channels, 8 * channels)
        self.middle_conv = DownConv(8 * channels, 16 * channels)
        self.up_conv1 = UpConv(16 * channels, 8 * channels)
        self.up_conv2 = UpConv(8 * channels, 4 * channels)
        self.up_conv3 = UpConv(4 * channels, 2 * channels)
        self.up_conv4 = UpConv(2 * channels, channels)
        self.last_conv = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.middle_conv(x4)
        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)
        return self.last_conv(u4)