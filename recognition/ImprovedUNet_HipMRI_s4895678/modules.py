"""Model components for HipMRI prostate segmentation.

Only PyTorch is imported here (no NumPy), as required by the task sheet.
Baseline: the 2D U-Net of Ronneberger et al. (2015), U-Net: Convolutional
Networks for Biomedical Image Segmentation, MICCAI.
"""
import torch
import torch.nn as nn


class DoubleConv2D(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """Standard 2D U-Net. Input [B,1,H,W] -> logits [B,num_classes,H,W].

    H and W must be divisible by 2**depth (256 is fine for depth=4).
    """

    def __init__(self, in_channels=1, num_classes=6, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]   # e.g. 32,64,128,256,512

        # Encoder: DoubleConv blocks, halving the image between them
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.enc.append(DoubleConv2D(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv2D(chs[-2], chs[-1])

        # Decoder: upsample, concatenate the skip connection, DoubleConv
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.dec.append(DoubleConv2D(c * 2, c))

        self.head = nn.Conv2d(chs[0], num_classes, 1)     # 1x1 conv -> class scores

    def forward(self, x):
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)          # saved for the skip connection
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)
