import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = F.relu(self.conv1(x))
        skip = self.conv2(skip)
        x = F.relu(x + skip)
        return x


class DeepRoutingUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, encoder_name='efficientnet-b1', encoder_weights=None):
        super().__init__()

        # EfficientNet encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )

        # Extract encoder output channels
        enc_channels = self.encoder.out_channels  # [C0, C1, C2, C3, C4, C5]

        self.center = nn.Conv2d(enc_channels[-1], enc_channels[-1], kernel_size=3, padding=1)

        self.decoder4 = DecoderResidualBlock(enc_channels[-1], enc_channels[-2], 256)
        self.decoder3 = DecoderResidualBlock(256, enc_channels[-3], 128)
        self.decoder2 = DecoderResidualBlock(128, enc_channels[-4], 64)
        self.decoder1 = DecoderResidualBlock(64, enc_channels[-5], 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)  # [C0, C1, C2, C3, C4, C5]

        x = self.center(features[-1])
        x = self.decoder4(x, features[-2])
        x = self.decoder3(x, features[-3])
        x = self.decoder2(x, features[-4])
        x = self.decoder1(x, features[-5])

        return self.final_conv(x)
