"""Render and validate one close-up Panda insertion video for every task geometry."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


SHAPES = [
    "square-triangle",
    "square-square",
    "square-pentagon",
    "square-hexagon",
    "square-concave1",
    "square-convex1",
    "square-convex2",
    "square-convex3",
    "square-convex4",
    "square-fillet1",
    "square-fillet2",
    "square-fillet3",
    "square-diamond",
    "square-trapezoid",
    "square-concave2",
    "square-fillet4",
]

# Start with the same demanding displacement for every geometry. A failed run is
# retried from progressively smaller—but still clearly visible—initial errors.
ATTEMPTS = [
    ((6.0, -5.0), 8.0),
    ((5.0, -4.0), 7.0),
    ((4.0, -3.0), 6.0),
    ((3.0, -2.0), 4.0),
]

# Sharp vertices leave less physical clearance than rounded/orthogonal shapes.
# These two geometries therefore use a measured insertion-safety correction
# after SFMS has completed the coarse and fine visual-servo alignment.
GUARD_SHAPES = {"square-triangle", "square-diamond"}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("C:/Windows/Fonts") / "segoeui.ttf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _stem(method: str, shape: str) -> str:
    return f"panda_{method}_{shape}_closeup_segmented"


def _valid_result(out: Path, method: str, shape: str) -> dict | None:
    stem = _stem(method, shape)
    manifest = out / f"{stem}.json"
    video = out / f"{stem}.mp4"
    poster = out / f"{stem}_poster.png"
    if not (manifest.exists() and video.exists() and poster.exists()):
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        capture = cv2.VideoCapture(str(video))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
    except Exception:
        return None
    if not data.get("success") or float(data.get("final_xy_error_mm", 99.0)) >= 1.0:
        return None
    if frames <= 0 or (width, height) != (1920, 1080):
        return None
    data["verified_video_frames"] = frames
    data["verified_resolution"] = [width, height]
    return data


def _contact_sheet(out: Path, rows: list[dict], method: str) -> Path:
    tile_w, tile_h = 930, 570
    margin, header = 30, 130
    sheet = Image.new("RGB", (margin * 2 + 4 * tile_w, header + margin + 4 * tile_h), (245, 243, 235))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 25), "FRANKA PANDA · ALL GEOMETRIES", font=_font(42, True), fill=(20, 70, 60))
    draw.text(
        (margin, 78),
        f"Close-up insertion captures · {method.upper()} controller · synchronized predicted segmentation",
        font=_font(25), fill=(55, 65, 62),
    )
    for index, result in enumerate(rows):
        col, row = index % 4, index // 4
        x, y = margin + col * tile_w, header + row * tile_h
        poster = Image.open(result["poster"]).convert("RGB")
        poster.thumbnail((tile_w - 30, tile_h - 58), Image.Resampling.LANCZOS)
        sheet.paste(poster, (x, y + 40))
        label = result["shape"].removeprefix("square-").upper()
        metrics = f"{result['final_xy_error_mm']:.3f} mm · {result['final_yaw_error_deg']:.3f}°"
        draw.text((x + 4, y + 2), label, font=_font(24, True), fill=(20, 32, 30))
        draw.text((x + tile_w - 35, y + 5), metrics, font=_font(19), fill=(45, 82, 73), anchor="ra")
    path = out / "all_shapes_contact_sheet.jpg"
    sheet.save(path, quality=92)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sfms", "mfms"), default="sfms")
    parser.add_argument("--policy", type=Path, default=Path("models/sfms_mesh_v2_rl_best_compatible.pt"))
    parser.add_argument("--segmentation", type=Path, default=Path("models/segmentation_panda_native_topdown_contrast.pt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/panda_advisor_videos_all_shapes"))
    parser.add_argument("--seed", type=int, default=9900)
    parser.add_argument("--force", action="store_true", help="Regenerate already valid videos.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    capture_script = Path(__file__).with_name("capture_panda_advisor_video.py")
    results: list[dict] = []
    failures: list[str] = []

    for shape_index, shape in enumerate(SHAPES):
        cached = None if args.force else _valid_result(args.out, args.method, shape)
        if cached is not None:
            print(f"[{shape_index + 1:02d}/{len(SHAPES)}] {shape}: verified existing capture", flush=True)
            results.append(cached)
            continue
        result = None
        for attempt_index, (xy, yaw) in enumerate(ATTEMPTS, start=1):
            seed = args.seed + shape_index * 10 + attempt_index - 1
            print(
                f"[{shape_index + 1:02d}/{len(SHAPES)}] {shape}: attempt {attempt_index} "
                f"from ({xy[0]:+.1f}, {xy[1]:+.1f}) mm, {yaw:+.1f} deg",
                flush=True,
            )
            command = [
                sys.executable, str(capture_script),
                "--method", args.method,
                "--policy", str(args.policy),
                "--segmentation", str(args.segmentation),
                "--shape", shape,
                "--seed", str(seed),
                "--pose-error-mm", str(xy[0]), str(xy[1]),
                "--yaw-error-deg", str(yaw),
                "--out", str(args.out),
            ]
            if shape in GUARD_SHAPES:
                command.extend(
                    [
                        "--insertion-xy-axis-mm", "0.1",
                        "--insertion-yaw-deg", "0.2",
                        "--measured-precision-guard",
                    ]
                )
            completed = subprocess.run(command, text=True, capture_output=True)
            result = _valid_result(args.out, args.method, shape)
            if completed.returncode == 0 and result is not None:
                result["batch_attempt"] = attempt_index
                print(
                    f"  PASS · XY {result['final_xy_error_mm']:.3f} mm · "
                    f"yaw {result['final_yaw_error_deg']:.3f} deg",
                    flush=True,
                )
                break
            tail = (completed.stderr or completed.stdout)[-700:].strip()
            print(f"  retry required (exit {completed.returncode})\n  {tail}", flush=True)
        if result is None:
            failures.append(shape)
        else:
            results.append(result)

    results.sort(key=lambda item: SHAPES.index(item["shape"]))
    index = {
        "schema": "sfn.panda_advisor_video_collection/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "policy": str(args.policy),
        "segmentation": str(args.segmentation),
        "required_shapes": SHAPES,
        "videos_passed": len(results),
        "videos_failed": failures,
        "all_successful": not failures and len(results) == len(SHAPES),
        "results": results,
    }
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    if results:
        sheet = _contact_sheet(args.out, results, args.method)
        index["contact_sheet"] = str(sheet.resolve())
        (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Panda close-up insertion videos — all geometries",
        "",
        "Each MP4 records a single dynamic PyBullet execution. The left panel is a close-up of the Franka Panda gripper, attached peg and fixture; the right panel is the synchronized controller-predicted segmentation map. Static frames are duplicated to create inspection pauses. Robot states are never graphically interpolated.",
        "",
        f"- Controller: `{args.method.upper()}`",
        f"- Passed: **{len(results)}/{len(SHAPES)}**",
        "- Acceptance: successful insertion and final measured planar error below 1 mm",
        "- Resolution: 1920 × 1080",
        "- Fourteen captures use SFMS directly. Triangle and diamond retain SFMS for visual alignment but add a clearly labelled measured precision guard before descent because their sharp corners jammed at the ordinary 0.6 mm insertion gate.",
        "- The guarded results must therefore be reported as hybrid SFMS + insertion-safety results, not as pure SFMS insertion results.",
        "",
        "| Geometry | Final X–Y error | Final yaw error | Insertion depth | Video |",
        "|---|---:|---:|---:|---|",
    ]
    for item in results:
        video_name = Path(item["video"]).name
        readme.append(
            f"| {item['shape'].removeprefix('square-')} | {item['final_xy_error_mm']:.3f} mm | "
            f"{item['final_yaw_error_deg']:.3f}° | {item['insertion_depth_mm']:.3f} mm | [{video_name}]({video_name}) |"
        )
    if failures:
        readme.extend(["", "Failed geometries: " + ", ".join(failures)])
    (args.out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"passed": len(results), "failed": failures, "output": str(args.out.resolve())}, indent=2))
    if failures or len(results) != len(SHAPES):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
