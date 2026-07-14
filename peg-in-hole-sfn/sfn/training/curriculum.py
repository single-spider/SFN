"""Dependency-ordered SFMS curriculum training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..config import CameraConfig, EnvironmentConfig
from ..evaluation.disturbance import ROBUSTNESS_PROFILES, DisturbanceConfig, DisturbedVirtualSensorNetwork
from ..models.vsn import VirtualSensorNetwork
from .train_sfms import SFMSTrainConfig, train_sfms


@dataclass(frozen=True)
class SFMSCurriculumStage:
    name: str
    updates: int
    rollout_steps: int = 64
    num_envs: int = 1
    xy_initial_range_mm: float = 10.0
    yaw_initial_range_deg: float = 10.0
    mask_source: str = "ground_truth"
    robustness_profile: str | None = None
    actor_lr: float = 3e-5
    critic_lr: float = 3e-4
    anchor_imitation_coef: float = 0.0

    def validate(self) -> None:
        if not self.name or any(character in self.name for character in "\\/:"):
            raise ValueError("curriculum stage name must be a non-empty filename-safe label")
        if self.updates <= 0 or self.rollout_steps <= 0 or self.num_envs <= 0:
            raise ValueError("updates, rollout_steps and num_envs must be positive")
        if self.mask_source not in {"ground_truth", "predicted"}:
            raise ValueError("mask_source must be ground_truth or predicted")
        if self.robustness_profile is not None and self.robustness_profile not in ROBUSTNESS_PROFILES:
            raise ValueError(f"unknown robustness profile: {self.robustness_profile}")


def stages_from_mapping(data: dict[str, Any]) -> list[SFMSCurriculumStage]:
    raw_stages = data.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("curriculum requires a non-empty stages list")
    allowed = set(SFMSCurriculumStage.__dataclass_fields__)
    stages = []
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, dict):
            raise ValueError(f"stages[{index}] must be a mapping")
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown stages[{index}] keys: {', '.join(unknown)}")
        stage = SFMSCurriculumStage(**raw)
        stage.validate()
        stages.append(stage)
    if len({stage.name for stage in stages}) != len(stages):
        raise ValueError("curriculum stage names must be unique")
    return stages


def run_sfms_curriculum(
    *,
    out_dir: str | Path,
    stages: list[SFMSCurriculumStage],
    seed: int,
    shapes: list[str],
    eval_shapes: list[str],
    environment: EnvironmentConfig,
    camera: CameraConfig,
    segmentation_path: str | Path | None,
    position_path: str | Path | None,
    orientation_path: str | Path | None,
    initial_policy_path: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    previous = Path(initial_policy_path) if initial_policy_path is not None else None
    results = []
    for index, stage in enumerate(stages):
        stage.validate()
        stage_dir = destination / f"{index + 1:02d}_{stage.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_env = replace(
            environment,
            xy_initial_range_mm=stage.xy_initial_range_mm,
            yaw_initial_range_deg=stage.yaw_initial_range_deg,
        )
        base_vsn = VirtualSensorNetwork.from_checkpoints(
            segmentation_path if stage.mask_source == "predicted" else None,
            position_path,
            orientation_path,
        )
        vsn = base_vsn
        if stage.robustness_profile is not None:
            profile = ROBUSTNESS_PROFILES[stage.robustness_profile]
            disturbance = DisturbanceConfig(**{**profile.to_dict(), "seed": seed + index * 10_000})
            vsn = DisturbedVirtualSensorNetwork(base_vsn, disturbance)
        checkpoint = stage_dir / "policy.pt"
        best = stage_dir / "best.pt"
        train_config = SFMSTrainConfig(
            updates=stage.updates,
            rollout_steps=stage.rollout_steps,
            num_envs=stage.num_envs,
            actor_lr=stage.actor_lr,
            critic_lr=stage.critic_lr,
            anchor_imitation_coef=stage.anchor_imitation_coef,
            eval_every=max(1, stage.updates // 10),
            eval_seed=seed + 900_000,
            checkpoint_every=max(1, stage.updates // 5),
            log_jsonl=str(stage_dir / "metrics.jsonl"),
            seed=seed + index,
            mask_source=stage.mask_source,
            device=device,
        )
        metrics = train_sfms(
            checkpoint,
            config=train_config,
            shapes=shapes,
            eval_shapes=eval_shapes,
            env_config=stage_env,
            camera_config=camera,
            segmentation_path=segmentation_path,
            position_path=position_path,
            orientation_path=orientation_path,
            vsn=vsn,
            initial_policy_path=previous,
            best_out=best,
        )
        results.append({"stage": asdict(stage), "checkpoint": str(checkpoint), "metrics": metrics})
        previous = best if best.is_file() else checkpoint
        (stage_dir / "resolved_stage.json").write_text(
            json.dumps({"stage": asdict(stage), "train_config": asdict(train_config)}, indent=2) + "\n",
            encoding="utf-8",
        )
    report = {"seed": seed, "stages": results, "selected_checkpoint": str(previous)}
    (destination / "curriculum_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
