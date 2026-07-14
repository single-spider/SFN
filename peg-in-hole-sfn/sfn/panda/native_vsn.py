"""Panda native body-ID mask VSN bridge."""

from __future__ import annotations

import numpy as np

from ..constants import MASK_PEG, MASK_SEAM, ORIENTATION_ANGLES_DEG, POSITION_GRID_SIZE
from ..geometry import decode_position, encode_position
from ..models.vsn import VirtualSensorNetwork, VSNOutput
from ..training.train_sfms import _require_torch
from .config import PandaConfig
from .template_pose import PandaTopdownTemplatePoseEstimator

# Fitted on Panda native-camera body-ID probe data at camera z=0.10 m.
# Features: peg cx, peg cy, base cx, base cy, peg-base dx, peg-base dy, bias.
PANDA_NATIVE_XY_WEIGHTS = np.asarray(
    [
        [0.00011300522344244679, 1.1409696354659914e-06],
        [-2.814926788863417e-07, -0.00012528462700457633],
        [6.0387142640298956e-05, 1.6822810336464023e-06],
        [-4.886914200281307e-07, -7.17522100202892e-05],
        [5.2618080802150164e-05, -5.413113981825723e-07],
        [2.0719874075686013e-07, -5.3532416983944235e-05],
        [-0.02159401857275287, 0.019380983636086344],
    ],
    dtype=np.float64,
)
PANDA_NATIVE_YAW_WEIGHTS = np.asarray([-0.05654058019874017, -0.06316664653726478], dtype=np.float64)


class PandaBodyIdGeometricVSN(VirtualSensorNetwork):
    """Decode Panda body-ID masks into SFMS/MFMS-style probability state.

    This is an explicit bridge for native PyBullet body-ID masks.  XY is usable
    after camera calibration; yaw remains weak and is reported with lower
    confidence because the current body-ID silhouette is not strongly
    yaw-observable.
    """

    def __init__(self):
        super().__init__()

    def forward(self, rgb=None, mask=None):  # noqa: D401
        if mask is None:
            raise ValueError("PandaBodyIdGeometricVSN requires a body-ID mask")
        torch, _ = _require_torch()
        mask_t = mask.long()
        device = mask_t.device
        b, h, w = mask_t.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        position_prob = torch.zeros((b, POSITION_GRID_SIZE, POSITION_GRID_SIZE), dtype=torch.float32, device=device)
        orientation_prob = torch.zeros((b, len(ORIENTATION_ANGLES_DEG)), dtype=torch.float32, device=device)
        dxy_values = []
        dyaw_values = []
        pos_conf = []
        yaw_conf = []
        for i in range(b):
            m = mask_t[i]
            peg = m == MASK_PEG
            base = m == MASK_SEAM
            if int(peg.sum()) == 0 or int(base.sum()) == 0:
                dx, dy, yaw = 0.0, 0.0, 0.0
                pc, yc = 0.01, 0.01
            else:
                pxx = xx[peg].float()
                pyy = yy[peg].float()
                bxx = xx[base].float()
                byy = yy[base].float()
                pcx = float(pxx.mean().detach().cpu())
                pcy = float(pyy.mean().detach().cpu())
                bcx = float(bxx.mean().detach().cpu())
                bcy = float(byy.mean().detach().cpu())
                feat = np.asarray([pcx, pcy, bcx, bcy, pcx - bcx, pcy - bcy, 1.0], dtype=np.float64)
                dx, dy = (feat @ PANDA_NATIVE_XY_WEIGHTS).tolist()
                x0 = (pxx - pxx.mean()).detach().cpu().numpy().astype(np.float64)
                y0 = (pyy - pyy.mean()).detach().cpu().numpy().astype(np.float64)
                mu20 = float(np.mean(x0 * x0))
                mu02 = float(np.mean(y0 * y0))
                mu11 = float(np.mean(x0 * y0))
                angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02) * 180.0 / np.pi
                angle = ((angle + 45.0) % 90.0) - 45.0
                yaw = float(np.asarray([angle, 1.0]) @ PANDA_NATIVE_YAW_WEIGHTS)
                pc, yc = 0.8, 0.35
            try:
                row, col = encode_position(float(dx), float(dy))
            except ValueError:
                row = int(np.clip(round(float(dy) * 1000.0) + 10, 0, POSITION_GRID_SIZE - 1))
                col = int(np.clip(round(-float(dx) * 1000.0) + 10, 0, POSITION_GRID_SIZE - 1))
            position_prob[i, row, col] = 1.0
            angles = np.asarray(ORIENTATION_ANGLES_DEG, dtype=np.float32)
            oi = int(np.argmin(np.abs(angles - float(yaw))))
            orientation_prob[i, oi] = 1.0
            dxy_values.append(decode_position(row, col))
            dyaw_values.append(float(ORIENTATION_ANGLES_DEG[oi]))
            pos_conf.append(pc)
            yaw_conf.append(yc)
        dxy = torch.as_tensor(dxy_values, dtype=torch.float32, device=device)
        dyaw = torch.as_tensor(dyaw_values, dtype=torch.float32, device=device)
        return VSNOutput(
            mask_logits=None,
            mask=mask_t,
            position_logits=position_prob.flatten(1),
            position_prob=position_prob,
            orientation_scores=orientation_prob,
            orientation_prob=orientation_prob,
            dxy_m=dxy,
            dyaw_deg=dyaw,
            position_confidence=torch.as_tensor(pos_conf, dtype=torch.float32, device=device),
            orientation_confidence=torch.as_tensor(yaw_conf, dtype=torch.float32, device=device),
        )


class PandaTopdownTemplateVSN(VirtualSensorNetwork):
    """VSN contract backed by calibrated, shape-aware mesh template matching."""

    def __init__(self, shape: str, panda_config: PandaConfig, width=500, height=400, fov_y_deg=35.0, segmentation=None):
        super().__init__(segmentation=segmentation)
        self.estimator = PandaTopdownTemplatePoseEstimator(shape, panda_config, width, height, fov_y_deg)

    def forward(self, rgb=None, mask=None):
        torch, _ = _require_torch()
        mask_logits = None
        if mask is None:
            if rgb is None:
                raise ValueError("rgb or mask is required")
            x = rgb.float()
            x = x / 255.0 if float(x.max()) > 3 else x
            mask_logits = self.segmentation(x)
            mask = torch.argmax(mask_logits, dim=1)
        device = mask.device
        pos_rows = []
        ori_rows = []
        dxy = []
        dyaw = []
        conf = []
        valid = []
        cells = np.arange(POSITION_GRID_SIZE, dtype=np.float32)
        center = (POSITION_GRID_SIZE - 1) / 2
        angles = np.asarray(ORIENTATION_ANGLES_DEG, dtype=np.float32)
        for m in mask.detach().cpu().numpy():
            xy, yaw, score, ok = self.estimator.estimate(m)
            mm = xy * 1000.0
            rows = cells[:, None] - center
            cols = center - cells[None, :]
            pos_rows.append(np.exp(-((rows - mm[1]) ** 2 + (cols - mm[0]) ** 2) / (2 * 0.45**2)))
            ori_rows.append(np.exp(-((angles - yaw) ** 2) / (2 * 0.75**2)))
            dxy.append(xy)
            dyaw.append(yaw)
            conf.append(score)
            valid.append(ok)
        pp = torch.as_tensor(np.asarray(pos_rows), dtype=torch.float32, device=device)
        pp /= pp.flatten(1).sum(1)[:, None, None].clamp_min(1e-8)
        op = torch.as_tensor(np.asarray(ori_rows), dtype=torch.float32, device=device)
        op /= op.sum(1)[:, None].clamp_min(1e-8)
        return VSNOutput(
            mask_logits,
            mask.long(),
            pp.flatten(1),
            pp,
            op,
            op,
            torch.as_tensor(np.asarray(dxy), dtype=torch.float32, device=device),
            torch.as_tensor(dyaw, dtype=torch.float32, device=device),
            torch.as_tensor(conf, dtype=torch.float32, device=device),
            torch.as_tensor(conf, dtype=torch.float32, device=device),
        )
