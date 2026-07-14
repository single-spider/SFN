from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .orientation import OrientationNet, decode_orientation_scores, orientation_model_from_config
from .position import PositionNet, decode_position_logits
from .segmentation import SegmentationModel


@dataclass
class VSNOutput:
    mask_logits: Tensor | None
    mask: Tensor
    position_logits: Tensor
    position_prob: Tensor
    orientation_scores: Tensor
    orientation_prob: Tensor
    dxy_m: Tensor
    dyaw_deg: Tensor
    position_confidence: Tensor
    orientation_confidence: Tensor
    valid: Tensor | None = None
    invalid_reason: tuple[str | None, ...] | None = None


class VirtualSensorNetwork(nn.Module):
    def __init__(
        self,
        segmentation: nn.Module | None = None,
        position: nn.Module | None = None,
        orientation: nn.Module | None = None,
    ):
        super().__init__()
        self.segmentation = segmentation or SegmentationModel()
        self.position = position or PositionNet()
        self.orientation = orientation or OrientationNet()

    def forward(self, rgb: Tensor | None = None, mask: Tensor | None = None) -> VSNOutput:
        if (rgb is None) == (mask is None):
            raise ValueError("Exactly one of rgb or mask is required")
        mask_logits = None
        if mask is None:
            x = rgb.float()
            x = x / 255.0 if x.max() > 3 else x
            mask_logits = self.segmentation(x)
            mask = torch.argmax(mask_logits, dim=1)
        mask_f = mask.float()
        encoded = (mask_f / 2.0).unsqueeze(1)
        pos_logits = self.position(encoded)
        dxy, pos_conf, pos_prob = decode_position_logits(pos_logits)
        continuous = getattr(self.position, "predict_continuous", None)
        if callable(continuous):
            dxy = continuous(encoded)
        orient_scores = self.orientation(encoded)
        dyaw, orient_conf, orient_prob = decode_orientation_scores(orient_scores)
        has_peg = (mask == 1).flatten(1).any(1)
        has_seam = (mask == 2).flatten(1).any(1)
        valid = has_peg & has_seam
        if not bool(valid.all()):
            invalid = ~valid
            pos_logits = pos_logits.clone()
            orient_scores = orient_scores.clone()
            pos_prob = pos_prob.clone()
            orient_prob = orient_prob.clone()
            dxy = dxy.clone()
            dyaw = dyaw.clone()
            pos_conf = pos_conf.clone()
            orient_conf = orient_conf.clone()
            pos_logits[invalid] = 0
            orient_scores[invalid] = 0
            pos_prob[invalid] = 0
            orient_prob[invalid] = 0
            dxy[invalid] = 0
            dyaw[invalid] = 0
            pos_conf[invalid] = 0
            orient_conf[invalid] = 0
        reasons = tuple(None if bool(ok) else "missing_peg_or_seam" for ok in valid.detach().cpu())
        return VSNOutput(
            mask_logits,
            mask.long(),
            pos_logits,
            pos_prob,
            orient_scores,
            orient_prob,
            dxy,
            dyaw,
            pos_conf,
            orient_conf,
            valid,
            reasons,
        )

    @classmethod
    def from_checkpoints(cls, segmentation_path=None, position_path=None, orientation_path=None, freeze: bool = True):
        """Build a VSN and load any provided state-dict checkpoints on CPU.

        Respect each checkpoint's model_config so checkpoints trained with
        non-default --base-channels can be loaded by evaluation/SFSS/SFMS.
        """
        from pathlib import Path

        from ..training.common import load_checkpoint_cpu

        def load_module(path, task: str):
            if path is None:
                return None
            ckpt = load_checkpoint_cpu(Path(path))
            cfg = ckpt.get("model_config", {})
            if task == "segmentation":
                module = SegmentationModel(**{k: v for k, v in cfg.items() if k in {"in_channels", "classes", "base"}})
            elif task == "position":
                if cfg.get("model_type") == "calibrated_geometric":
                    from .position import CalibratedGeometricPositionNet

                    module = CalibratedGeometricPositionNet(
                        cfg["feature_mean"],
                        cfg["feature_std"],
                        cfg["weights"],
                        grid_size=cfg.get("grid_size", 21),
                        temperature_mm=cfg.get("temperature_mm", 0.5),
                    )
                else:
                    module = PositionNet(**{k: v for k, v in cfg.items() if k in {"in_channels", "grid_size", "base"}})
            else:
                module = orientation_model_from_config(cfg)
            module.load_state_dict(ckpt["model_state_dict"])
            return module

        vsn = cls(
            segmentation=load_module(segmentation_path, "segmentation"),
            position=load_module(position_path, "position"),
            orientation=load_module(orientation_path, "orientation"),
        )
        if freeze:
            for p in vsn.parameters():
                p.requires_grad_(False)
            vsn.eval()
        return vsn
