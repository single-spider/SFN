from __future__ import annotations

from .unet import TinyUNet


class SegmentationModel(TinyUNet):
    def __init__(self, in_channels=3, classes=3, base=16):
        super().__init__(in_channels=in_channels, out_channels=classes, base=base)
