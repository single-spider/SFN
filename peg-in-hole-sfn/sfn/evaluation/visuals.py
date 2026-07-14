"""Visual artifact helpers for datasets and VSN predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..data.dataset import NPZDataset
from ..geometry import decode_orientation, decode_position

PALETTE = np.asarray(
    [
        [35, 35, 35],  # background
        [35, 210, 55],  # peg
        [230, 170, 40],  # seam
    ],
    dtype=np.uint8,
)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask.astype(np.int64), 0, 2)]


def overlay_mask(rgb_hwc: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = mask_to_rgb(mask)
    return np.clip((1 - alpha) * rgb_hwc + alpha * color, 0, 255).astype(np.uint8)


def _draw_label(img, text: str):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))


def _heatmap_img(target: np.ndarray, size=(250, 200)) -> Image.Image:
    arr = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    arr[..., :] = 30
    arr[target > 0] = [255, 40, 40]
    return Image.fromarray(arr, "RGB").resize(size, resample=Image.Resampling.NEAREST)


def render_sample_panel(sample: dict, prediction: dict | None = None):
    from PIL import Image, ImageDraw

    rgb = np.transpose(sample["rgb"], (1, 2, 0)).astype(np.uint8)
    mask = sample["mask"].astype(np.uint8)
    panels = [
        ("RGB", Image.fromarray(rgb, "RGB")),
        ("GT mask", Image.fromarray(mask_to_rgb(mask), "RGB")),
        ("Overlay", Image.fromarray(overlay_mask(rgb, mask), "RGB")),
        ("XY target", _heatmap_img(sample["position_target"])),
    ]
    if prediction is not None:
        panels.append(("Pred mask", Image.fromarray(mask_to_rgb(prediction["mask"]), "RGB")))
        panels.append(("Pred overlay", Image.fromarray(overlay_mask(rgb, prediction["mask"]), "RGB")))

    for label, img in panels:
        _draw_label(img, label)
    w, h = panels[0][1].size
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h + 42), (20, 20, 20))
    for i, (_, img) in enumerate(panels):
        canvas.paste(img, ((i % cols) * w, (i // cols) * h))
    draw = ImageDraw.Draw(canvas)
    pose = sample["pose_error"]
    row, col = np.unravel_index(int(np.argmax(sample["position_target"])), sample["position_target"].shape)
    dx, dy = decode_position(row, col)
    ori = decode_orientation(int(sample["orientation_index"]))
    text = f"shape={sample['shape_id']} pose=[{pose[0] * 1000:.1f},{pose[1] * 1000:.1f},{pose[2]:.1f}]mm/deg target_xy=[{dx * 1000:.1f},{dy * 1000:.1f}] ori={ori:.1f}"
    if prediction is not None:
        text += f" pred_xy=[{prediction['dxy_m'][0] * 1000:.1f},{prediction['dxy_m'][1] * 1000:.1f}] pred_yaw={prediction['dyaw_deg']:.1f}"
    draw.text((4, rows * h + 6), text, fill=(255, 255, 255))
    return canvas


def generate_dataset_visuals(
    dataset: str | Path,
    out_dir: str | Path,
    count: int = 4,
    segmentation_path: str | Path | None = None,
    position_path: str | Path | None = None,
    orientation_path: str | Path | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = NPZDataset(dataset)
    vsn = None
    if segmentation_path or position_path or orientation_path:
        import torch

        from ..models import VirtualSensorNetwork

        vsn = VirtualSensorNetwork.from_checkpoints(segmentation_path, position_path, orientation_path)
    paths = []
    for i in range(min(count, len(ds))):
        sample = ds[i]
        prediction = None
        if vsn is not None:
            import torch

            with torch.no_grad():
                if segmentation_path:
                    rgb = torch.as_tensor(sample["rgb"][None], dtype=torch.float32)
                    out = vsn(rgb=rgb)
                else:
                    mask = torch.as_tensor(sample["mask"][None], dtype=torch.long)
                    out = vsn(mask=mask)
            prediction = {
                "mask": out.mask[0].cpu().numpy().astype(np.uint8),
                "dxy_m": out.dxy_m[0].cpu().numpy(),
                "dyaw_deg": float(out.dyaw_deg[0].cpu()),
            }
        img = render_sample_panel(sample, prediction)
        path = out_dir / f"sample_{i:04d}.png"
        img.save(path)
        paths.append(path)
    return paths
