#!/usr/bin/env python
"""Record verified local PyBullet showcase telemetry for public replay.

This script uses the same bounded local service execution path as the browser
demo.  It records measured Panda joint and pose telemetry, not a browser-side
animation.  By default it creates one representative replay for each method;
pass ``--all-shapes`` to make the full 16-shape by 3-method replay archive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sfn.constants import ALL_EXPECTED_SHAPES
from sfn.showcase.schema import SessionCommand, SessionRequest
from sfn.showcase.service import SessionManager


ROOT = Path(__file__).resolve().parents[1]


def wait_for_completion(manager: SessionManager, session_id: str) -> None:
    """Drain events so a bounded session can finish without a WebSocket client."""
    while True:
        event = manager.take_event(session_id, timeout_s=1.0)
        if event is None:
            continue
        if event.event == "error":
            raise RuntimeError(event.message or "live Panda replay failed")
        if event.event == "episode_finished":
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/showcase/replays")
    parser.add_argument("--shape", default="square-concave2", choices=sorted(ALL_EXPECTED_SHAPES))
    parser.add_argument("--all-shapes", action="store_true", help="record all shapes for every method")
    parser.add_argument("--seed", type=int, default=9900)
    args = parser.parse_args()

    output = args.out if args.out.is_absolute() else ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    shapes = sorted(ALL_EXPECTED_SHAPES) if args.all_shapes else [args.shape]
    manager = SessionManager()
    index: list[dict[str, object]] = []
    for shape in shapes:
        for method in ("sfss", "sfms", "mfms"):
            request = SessionRequest(shape=shape, method=method, seed=args.seed)
            started = manager.create(request)
            manager.command(started.session_id, SessionCommand(command="start"))
            wait_for_completion(manager, started.session_id)
            # The finished event is emitted just before PyBullet cleanup; wait
            # briefly for the worker thread so the recording is immutable.
            for _ in range(100):
                try:
                    replay = manager.recording(started.session_id)
                    break
                except RuntimeError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("Panda replay worker did not finish cleanup")
            target = output / f"{shape}__{method}.json"
            target.write_text(replay.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8")
            index.append({"shape": shape, "method": method, "file": target.name, "frames": len(replay.frames)})
            print(f"recorded {shape} / {method}: {len(replay.frames)} frames")
            manager.command(started.session_id, SessionCommand(command="close"))
            time.sleep(0.05)
    (output / "index.json").write_text(json.dumps({"replays": index}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
