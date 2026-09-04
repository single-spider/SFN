"""Transcode Panda evidence MP4 files to browser-compatible VP8/WebM."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2


def transcode(source: Path, destination: Path) -> tuple[str, int, int]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    temporary = destination.with_suffix(".tmp.webm")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV VP8/WebM encoder is unavailable")
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        writer.release()
    if frames != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Frame mismatch for {source.name}: {frames} != {expected}")
    temporary.replace(destination)
    check = cv2.VideoCapture(str(destination))
    decoded = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if decoded != frames:
        raise RuntimeError(f"Validation failed for {destination.name}: {decoded} != {frames}")
    return destination.name, frames, destination.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sources = sorted(args.source.glob("panda_*.mp4"))
    if not sources:
        raise SystemExit(f"No Panda MP4 files found in {args.source}")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        jobs = {
            executor.submit(transcode, source, args.destination / source.with_suffix(".webm").name): source
            for source in sources
        }
        for index, future in enumerate(as_completed(jobs), start=1):
            name, frames, size = future.result()
            print(f"[{index:02d}/{len(sources)}] {name}: {frames} frames, {size / 1_048_576:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
