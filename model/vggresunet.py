import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

PROB_DROPOUT = 0.5

# VGG19 architecture: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
# M refers to max pooling operation.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.sample = None
        # Upsample / Downsample if the number of channels are not the same
        if in_channels != out_channels:
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    # Apply double conv with batch norm, relu and residual connections
    def forward(self, x):
        id = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.sample is not None:
            id = self.sample(id)

        y += id
        y = self.relu(y)

        return y

class QuadrupleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(QuadrupleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.sample = None
        # Upsample / Downsample if the number of channels are not the same
        if in_channels != out_channels:
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    # Apply quadruple conv with batch norm, relu and residual connections
    def forward(self, x):
        id = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        y = self.conv4(y)
        y = self.bn4(y)

        if self.sample is not None:
            id = self.sample(id)

        y += id
        y = self.relu(y)

        return y

class DoubleConvBottleneck(nn.Module):
    # Like DoubleConv but with dropout 
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.drop1 = nn.Dropout(p=PROB_DROPOUT)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop2 = nn.Dropout(p=PROB_DROPOUT)
        self.relu = nn.ReLU(inplace=True)

        self.sample = None
        # Upsample / Downsample if the number of channels are not the same
        if in_channels != out_channels:
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    # Apply double conv with batch norm, relu and residual connections
    def forward(self, x):
        id = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.sample is not None:
            id = self.sample(id)

        y += id
        y = self.relu(y)
        y = self.drop2(y)

        return y

class VGGRESUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features_double=[64, 128], features_quadruple=[256, 512, 512],
    ):
        super(VGGRESUNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part of VGGUNET
        for feature in features_double:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in features_quadruple:
            self.downs.append(QuadrupleConv(in_channels, feature))
            in_channels = feature

        # Decoder part of VGGUNET
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

        # Bottom part with dropouts
        self.bottleneck = DoubleConvBottleneck(features_quadruple[-1], features_quadruple[-1]*2)
        # Final conv to get one final channel
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            # Apply encoder
            x = down(x)
            # Save input for the skip connections
            skip_connections.append(x)
            x = self.pool(x)

        # Apply bottom part with dropouts
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concat input with the skip connections
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Apply decoder
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)