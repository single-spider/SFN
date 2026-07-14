from __future__ import annotations

from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=16):
        super().__init__()
        self.enc = ConvBlock(in_channels, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        return self.out(self.enc(x))
