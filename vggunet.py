import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from constants import PROB_DROPOUT

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class QuadrupleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(QuadrupleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=PROB_DROPOUT),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=PROB_DROPOUT),
        )

    def forward(self, x):
        return self.conv(x)

class VGGUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features_double=[64, 128], features_quadruple=[256, 512, 512],
    ):
        super(VGGUNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of VGGUNET
        for feature in features_double:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in features_quadruple:
            self.downs.append(QuadrupleConv(in_channels, feature))
            in_channels = feature

        # Up part of VGGUNET
        in_channels = in_channels*2
        for feature in reversed(features_quadruple):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(QuadrupleConv(feature*2, feature))
            in_channels = feature

        for feature in reversed(features_double):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            in_channels = feature

        self.bottleneck = DoubleConvBottleneck(features_quadruple[-1], features_quadruple[-1]*2)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
