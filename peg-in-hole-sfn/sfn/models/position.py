from __future__ import annotations

import torch
from torch import nn

from ..geometry import decode_position


class PositionNet(nn.Module):
    """Learned x-y offset classifier.

    The previous implementation analytically decoded the peg centroid from the
    ground-truth mask.  That was useful as a geometry smoke test, but it made a
    freshly initialized model report perfect validation metrics.  This version
    is intentionally a normal trainable classifier: it maps the encoded mask to
    21x21 position logits without hard-coding the centroid calculation.
    """

    def __init__(self, in_channels=1, grid_size=21, base=16):
        super().__init__()
        self.grid_size = int(grid_size)

        b = int(base)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=5, padding=2),
            nn.GroupNorm(4 if b >= 4 else 1, b),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, b, kernel_size=3, padding=1),
            nn.GroupNorm(4 if b >= 4 else 1, b),
            nn.SiLU(inplace=True),
        )

        # Position labels factor into independent row/column offsets.  Keep
        # high-resolution 1-D projections so one-pixel (1 mm) shifts are not
        # destroyed by early strided pooling, then combine row/column logits
        # into the 441-way classifier expected by the training/eval code.
        self.row_pool = nn.AdaptiveAvgPool1d(96)
        self.col_pool = nn.AdaptiveAvgPool1d(120)
        self.row_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b * 96, b * 8),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(b * 8, self.grid_size),
        )
        self.col_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b * 120, b * 8),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(b * 8, self.grid_size),
        )

    def forward(self, x):
        feat = self.features(x)
        row_signal = feat.mean(dim=3)
        col_signal = feat.mean(dim=2)
        row_logits = self.row_head(self.row_pool(row_signal))
        col_logits = self.col_head(self.col_pool(col_signal))
        logits = row_logits[:, :, None] + col_logits[:, None, :]
        return logits.reshape(x.shape[0], self.grid_size * self.grid_size)


class CalibratedGeometricPositionNet(nn.Module):
    """Sub-cell XY baseline fitted from semantic-region image statistics."""

    def __init__(self, feature_mean, feature_std, weights, grid_size=21, temperature_mm=0.5):
        super().__init__()
        self.grid_size = int(grid_size)
        self.temperature_mm = float(temperature_mm)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    @staticmethod
    def _region_features(mask, yy, xx):
        values = []
        for region in (mask == 1, mask == 2, mask > 0):
            r = region.float()
            mass = r.flatten(1).sum(1).clamp_min(1.0)
            mx = (r * xx).flatten(1).sum(1) / mass
            my = (r * yy).flatten(1).sum(1) / mass
            sx = torch.sqrt(((r * (xx - mx[:, None, None]) ** 2).flatten(1).sum(1) / mass).clamp_min(0.0))
            sy = torch.sqrt(((r * (yy - my[:, None, None]) ** 2).flatten(1).sum(1) / mass).clamp_min(0.0))
            values.extend((mx, my, sx, sy, mass / float(mask.shape[-2] * mask.shape[-1])))
        return torch.stack(values, dim=1)

    def predict_continuous(self, x):
        mask = torch.round(x[:, 0] * 2.0).long()
        h, w = mask.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        raw = self._region_features(mask, yy, xx)
        z = (raw - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        design = torch.cat((z, z * z, torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)), dim=1)
        return (design @ self.weights) / 1000.0

    def forward(self, x):
        dxy_mm = self.predict_continuous(x) * 1000.0
        cells = torch.arange(self.grid_size, device=x.device, dtype=x.dtype)
        # decode_position: dx=(center-col) mm, dy=(row-center) mm.
        center = (self.grid_size - 1) / 2.0
        rows = cells - center
        cols = center - cells
        dist2 = (rows[None, :, None] - dxy_mm[:, 1, None, None]) ** 2 + (
            cols[None, None, :] - dxy_mm[:, 0, None, None]
        ) ** 2
        return (-dist2 / max(self.temperature_mm**2, 1e-6)).reshape(x.shape[0], -1)


def decode_position_logits(logits):
    if logits.dim() == 2:
        prob_flat = torch.softmax(logits, dim=1)
        flat = torch.argmax(prob_flat, dim=1)
        grid = int(round(logits.shape[1] ** 0.5))
        conf = prob_flat.max(dim=1).values
        prob = prob_flat.reshape(logits.shape[0], grid, grid)
    else:
        prob = torch.softmax(logits, dim=1)[:, 1]
        flat = torch.argmax(prob.flatten(1), dim=1)
        grid = logits.shape[-1]
        conf = prob.flatten(1).max(dim=1).values
    rows = flat // grid
    cols = flat % grid
    vals = [decode_position(int(r), int(c), grid_size=grid) for r, c in zip(rows.cpu(), cols.cpu(), strict=False)]
    return torch.tensor(vals, dtype=logits.dtype, device=logits.device), conf, prob
