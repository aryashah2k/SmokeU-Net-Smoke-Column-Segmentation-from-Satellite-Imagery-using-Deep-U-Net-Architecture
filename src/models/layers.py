import torch.nn as nn

class Conv3k(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv1(x)

class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3k(channels_in, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(),
            Conv3k(channels_out, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class DownConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(channels_in, channels_out)
        )

    def forward(self, x):
        return self.encoder(x)

class UpConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(channels_in, channels_in // 2, kernel_size=1, stride=1)
        )
        self.decoder = DoubleConv(channels_in, channels_out)

    def forward(self, x1, x2):
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.decoder(x)