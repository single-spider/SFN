# Hardware readiness (software boundary)

This package is **not a robot driver or a safety-rated controller**. It prepares and tests the software boundary before a supervised hardware integration. Vendor emergency stops, collision monitoring, force/torque limits, speed limits, and a trained operator remain mandatory.

## Architecture

- `sfn.sim2real.interfaces` defines vendor-neutral Cartesian commands, minimal robot feedback, and the `CommandSink` protocol. A hardware adapter should be the only module that imports a vendor SDK.
- `CommandSafetyGate` checks confidence, frame age, finite/action magnitude, command rate, projected Cartesian workspace, and cumulative translation/rotation budgets.
- `GuardedCommandSession` starts in dry-run by default, can be explicitly armed, enters `REACQUIRE` after stale/low-confidence observations, and has a terminal software stop state.
- `sfn.sim2real.replay` reads naturally ordered image folders or videos as RGB frames with timestamps.
- `sfn.sim2real.annotations` exchanges indexed PNG masks and polygon-based, single-category COCO-style JSON. COCO RLE is intentionally left to a future optional `pycocotools` adapter.

## Offline workflow

```powershell
python scripts/replay_sim2real.py recordings/run_001 --image-fps 15 --output artifacts/run_001.frames.jsonl
python scripts/convert_sim2real_annotations.py export-coco labels/masks labels/annotations.json
python scripts/convert_sim2real_annotations.py import-coco labels/annotations.json labels/restored
pytest tests/test_sim2real_*.py
```

Feed replay frames through the exact perception/policy callback planned for hardware, but use `RecordingCommandSink`. Review every proposed command and safety decision. Tune limits in metres, radians, and seconds from the workcell risk assessment—not from simulation convenience.

## Adapter checklist

1. Convert the declared command frame to the driver's frame explicitly; never rely on an implicit base/tool convention.
2. Populate `RobotState` from measured state, not the last commanded pose.
3. Make `send()` blocking or acknowledged so `gate.commit()` represents an accepted command. Driver rejection must not be silently swallowed.
4. Map `stop(reason)` to a safe vendor-supported deceleration/hold. It does not replace an emergency stop.
5. Use a monotonic clock for frame timestamps, `issued_at`, and `now`; synchronize camera and control acquisition.
6. Keep hardware adapters outside `sfn/panda/` unless they are deliberately Panda-specific. The contracts here are robot-agnostic.

## Required staged validation

- Unit tests and recorded-media replay pass.
- Dry-run on live camera with motors disabled; stale/disconnected camera forces `REACQUIRE`.
- Armed free-space motion at reduced speed with conservative workspace and cumulative budgets.
- Deliberately test low confidence, frozen frames, non-monotonic timing, oversized actions, rate jumps, workspace edges, and budget exhaustion.
- Verify physical E-stop, protective stop, operator line of sight, fixture clearance, and independent force/speed limits before contact.
- Reset cumulative budgets only as an explicit operator action after checking the physical pose.

## Known boundary

The software gate predicts workspace from `position + translation`; it does not perform kinematics, collision checking, latency compensation, contact detection, or validate orientation-dependent tool geometry. Those belong in the hardware adapter and safety-rated robot/workcell controls.
