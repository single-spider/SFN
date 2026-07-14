"""Relative peg/hole orientation estimators."""

from __future__ import annotations

import torch
from torch import nn

from ..constants import ORIENTATION_ANGLES_DEG


def _symmetry_aware_axis_deg(peg: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
    """Second/fourth central-moment angle for 2-fold and 4-fold silhouettes."""
    mass = peg.flatten(1).sum(1).clamp_min(1e-6)
    cx = (peg * xx).flatten(1).sum(1) / mass
    cy = (peg * yy).flatten(1).sum(1) / mass
    x0, y0 = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    mu20 = (peg * x0 * x0).flatten(1).sum(1) / mass
    mu02 = (peg * y0 * y0).flatten(1).sum(1) / mass
    mu11 = (peg * x0 * y0).flatten(1).sum(1) / mass
    angle2 = -0.5 * torch.atan2(2.0 * mu11, mu20 - mu02)
    anisotropy = torch.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11**2) / (mu20 + mu02).clamp_min(1e-6)
    z = torch.complex(x0, y0)
    moment4 = (peg * z**4).flatten(1).sum(1) / mass
    angle4 = -0.25 * torch.angle(moment4) + torch.pi / 4
    angle = torch.where(anisotropy < 0.02, angle4, angle2)
    angle = torch.remainder(angle + torch.pi / 4, torch.pi / 2) - torch.pi / 4
    return angle * 180.0 / torch.pi


class OrientationNet(nn.Module):
    """Learn yaw from the joint peg-and-visible-seam mask.

    Unlike the former principal-axis formula, this network receives both
    semantic regions and can learn their *relative* geometry.  Global average
    pooling keeps the checkpoint independent of camera crop dimensions.
    """

    def __init__(self, in_channels=1, angles=ORIENTATION_ANGLES_DEG, base=16):
        super().__init__()
        self.angles = list(angles)
        b = int(base)
        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), b, 5, stride=2, padding=2),
            nn.BatchNorm2d(b),
            nn.SiLU(),
            nn.Conv2d(b, b * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.SiLU(),
            nn.Conv2d(b * 2, b * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.SiLU(),
            nn.Conv2d(b * 4, b * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b * 4 * 16, b * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(b * 4, len(self.angles)),
        )
        # Begin from the calibrated geometric solution and learn shape/seam
        # residuals rather than rediscovering image-axis sign from scratch.
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "angle_values", torch.tensor(self.angles, dtype=torch.float32).reshape(1, -1), persistent=False
        )

    def forward(self, x):
        residual = self.classifier(self.features(x))
        peg = torch.clamp(1.0 - torch.abs(x[:, 0] - 0.5) * 4.0, 0.0, 1.0)
        b, h, w = peg.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        task_angle = _symmetry_aware_axis_deg(peg, yy, xx)
        geometric_logits = -((self.angle_values.to(x.device, x.dtype) - task_angle[:, None]) ** 2)
        return geometric_logits * self.logit_scale.exp().clamp(0.1, 10.0) + residual


class RelativeOrientationNet(nn.Module):
    """Siamese peg-versus-seam yaw classifier.

    The legacy :class:`OrientationNet` consumes the semantic mask as one scalar
    image, so its learned path is free to solve yaw from the peg alone.  This
    model instead splits the mask into peg and visible-hole/seam silhouettes,
    encodes both with *the same* convolutional tower, and exposes only ordered
    difference and multiplicative-correlation features to the classifier.  A
    prediction therefore has to be formed from peg-versus-hole evidence rather
    than from a concatenated single-image representation.

    ``OrientationNet`` is intentionally retained unchanged for historical
    checkpoints.  New checkpoints identify this architecture with
    ``model_type=relative_siamese_correlation``.
    """

    model_type = "relative_siamese_correlation"

    def __init__(self, in_channels=1, angles=ORIENTATION_ANGLES_DEG, base=16):
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError("RelativeOrientationNet expects the encoded one-channel semantic mask")
        self.angles = list(angles)
        b = int(base)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, b, 5, stride=2, padding=2),
            nn.BatchNorm2d(b),
            nn.SiLU(),
            nn.Conv2d(b, b * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.SiLU(),
            nn.Conv2d(b * 2, b * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.SiLU(),
            nn.Conv2d(b * 4, b * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # There are deliberately no peg-only or seam-only skip features here.
        # Signed difference preserves which member of the pair is the peg;
        # the Hadamard term is a learned channel-wise correlation.
        pair_features = b * 4 * 4 * 4 * 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pair_features, b * 8),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(b * 8, len(self.angles)),
        )

    @staticmethod
    def split_semantic_mask(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable peg and visible-seam occupancy channels.

        VSN masks are encoded as background=0, peg=0.5, seam=1.  Triangular
        membership functions preserve exact binary channels for hard masks and
        remain usable with soft/noisy masks.
        """

        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("orientation input must have shape [batch, 1, height, width]")
        peg = torch.clamp(1.0 - 4.0 * torch.abs(x - 0.5), 0.0, 1.0)
        seam = torch.clamp(1.0 - 2.0 * torch.abs(x - 1.0), 0.0, 1.0)
        return peg, seam

    def pair_features(self, x: torch.Tensor) -> torch.Tensor:
        peg, seam = self.split_semantic_mask(x)
        peg_features = self.encoder(peg)
        seam_features = self.encoder(seam)
        return torch.cat((peg_features - seam_features, peg_features * seam_features), dim=1)

    def forward(self, x):
        return self.classifier(self.pair_features(x))


def orientation_model_from_config(config: dict | None = None) -> nn.Module:
    """Instantiate old and new orientation checkpoints without migration."""

    cfg = dict(config or {})
    kwargs = {"in_channels": cfg.get("in_channels", 1), "base": cfg.get("base", 16)}
    angles = cfg.get("angles", None) or cfg.get("orientation_angles_deg", None)
    if angles is not None:
        kwargs["angles"] = angles
    if cfg.get("model_type") == RelativeOrientationNet.model_type:
        return RelativeOrientationNet(**kwargs)
    return OrientationNet(**kwargs)


class GeometricOrientationNet(nn.Module):
    """Legacy peg-only principal-axis baseline, retained for explicit ablation."""

    def __init__(self, in_channels=1, angles=ORIENTATION_ANGLES_DEG, base=16):
        super().__init__()
        self.angles = list(angles)
        self.logit_scale = nn.Parameter(torch.tensor(1.5))
        self.register_buffer(
            "angle_values", torch.tensor(self.angles, dtype=torch.float32).reshape(1, -1), persistent=False
        )

    def forward(self, x):
        b, _, h, w = x.shape
        peg = torch.clamp(1.0 - torch.abs(x[:, 0] - 0.5) * 4.0, 0.0, 1.0)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        angle_deg = _symmetry_aware_axis_deg(peg, yy, xx)
        dist2 = (self.angle_values.to(x.device, x.dtype) - angle_deg[:, None]) ** 2
        return -dist2 * self.logit_scale.exp().clamp(0.1, 50.0)


def decode_orientation_scores(scores, angles=ORIENTATION_ANGLES_DEG):
    prob = torch.softmax(scores, dim=1)
    idx = torch.argmax(prob, dim=1)
    angle_tensor = torch.tensor(list(angles), dtype=scores.dtype, device=scores.device)
    return angle_tensor[idx], prob.max(dim=1).values, prob
