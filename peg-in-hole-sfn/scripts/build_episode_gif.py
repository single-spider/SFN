#!/usr/bin/env python
"""Build an animated GIF from ordered evaluation frame PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True, help="Directory containing step_*.png files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=700)
    args = parser.parse_args()
    paths = sorted(args.frames.glob("step_*.png"))
    if not paths:
        raise SystemExit(f"No step PNGs found under {args.frames}")
    frames = [Image.open(path).convert("RGB") for path in paths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Wrote {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
