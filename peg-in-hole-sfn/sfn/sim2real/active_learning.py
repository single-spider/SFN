"""Deterministic active-learning and pseudo-label review utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def entropy_uncertainty(probabilities: np.ndarray, *, class_axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Return normalized predictive entropy for every sample.

    ``probabilities`` normally has shape ``(N, C, H, W)``.  Spatial dimensions
    are averaged, so the result always has shape ``(N,)``.  Values are in
    ``[0, 1]`` (zero for a certain prediction, one for a uniform prediction).
    Inputs are renormalized along the class axis to tolerate small numerical
    drift or non-negative confidence maps.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim < 2:
        raise ValueError("probabilities must include sample and class dimensions")
    axis = int(class_axis) % probs.ndim
    if axis == 0:
        raise ValueError("class_axis cannot be the sample axis")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = probs.sum(axis=axis, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("each prediction must have positive probability mass")
    probs = probs / total
    entropy = -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=axis)
    classes = probs.shape[axis]
    if classes > 1:
        entropy /= np.log(classes)
    reduce_axes = tuple(range(1, entropy.ndim))
    return entropy.mean(axis=reduce_axes) if reduce_axes else entropy


def margin_uncertainty(probabilities: np.ndarray, *, class_axis: int = 1) -> np.ndarray:
    """Return ``1 - (largest probability - second largest)`` per sample."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim < 2:
        raise ValueError("probabilities must include sample and class dimensions")
    axis = int(class_axis) % probs.ndim
    if axis == 0 or probs.shape[axis] < 2:
        raise ValueError("class_axis must identify at least two classes")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    probs = probs / np.maximum(probs.sum(axis=axis, keepdims=True), 1e-12)
    ordered = np.sort(probs, axis=axis)
    margin = np.take(ordered, -1, axis=axis) - np.take(ordered, -2, axis=axis)
    reduce_axes = tuple(range(1, margin.ndim))
    value = 1.0 - margin
    return value.mean(axis=reduce_axes) if reduce_axes else value


def _unit_interval(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def _feature_distance(features: np.ndarray) -> np.ndarray:
    vectors = np.asarray(features, dtype=np.float64)
    if vectors.ndim < 2:
        raise ValueError("features must have shape (samples, ...)")
    vectors = vectors.reshape(vectors.shape[0], -1)
    if np.any(~np.isfinite(vectors)):
        raise ValueError("features must be finite")
    # Standardized Euclidean distance works for both scalar descriptors and
    # learned embeddings, including meaningful zero vectors.
    scale = vectors.std(axis=0, keepdims=True)
    vectors = (vectors - vectors.mean(axis=0, keepdims=True)) / np.where(scale > 1e-12, scale, 1.0)
    delta = vectors[:, None, :] - vectors[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, 0.0)
    return distance


def select_active_learning_indices(
    uncertainty: Sequence[float] | np.ndarray,
    budget: int,
    *,
    features: np.ndarray | None = None,
    diversity_weight: float = 0.5,
    seed: int = 0,
) -> list[int]:
    """Greedily select uncertain and diverse samples with deterministic ties.

    The first item is the most uncertain.  Later items maximize a weighted sum
    of normalized uncertainty and distance to the nearest selected feature.
    ``seed`` only resolves exact ties, making repeated calls reproducible while
    allowing a caller to vary otherwise equivalent selections.
    """

    scores = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(scores)):
        raise ValueError("uncertainty values must be finite")
    if int(budget) < 0:
        raise ValueError("budget must be non-negative")
    if not 0.0 <= float(diversity_weight) <= 1.0:
        raise ValueError("diversity_weight must be in [0, 1]")
    count = scores.size
    budget = min(int(budget), count)
    if budget == 0:
        return []
    distance = None
    if features is not None:
        if np.asarray(features).shape[0] != count:
            raise ValueError("features and uncertainty must have the same sample count")
        distance = _feature_distance(np.asarray(features))

    normalized = _unit_interval(scores)
    rng = np.random.default_rng(int(seed))
    tie_break = rng.random(count)
    selected: list[int] = []
    available = np.ones(count, dtype=bool)
    while len(selected) < budget:
        if not selected or distance is None or diversity_weight == 0:
            merit = normalized.copy()
        else:
            nearest = distance[:, selected].min(axis=1)
            diversity = _unit_interval(nearest)
            merit = (1.0 - diversity_weight) * normalized + diversity_weight * diversity
        merit[~available] = -np.inf
        best_value = float(np.max(merit))
        tied = np.flatnonzero(available & np.isclose(merit, best_value, rtol=0.0, atol=1e-12))
        choice = int(tied[np.argmax(tie_break[tied])])
        selected.append(choice)
        available[choice] = False
    return selected


def select_active_learning_frames(
    probabilities: np.ndarray,
    budget: int,
    *,
    embeddings: np.ndarray | None = None,
    frame_ids: Sequence[str] | None = None,
    method: str = "entropy",
    diversity_weight: float = 0.5,
    seed: int = 0,
    class_axis: int = 1,
) -> list[int] | list[str]:
    """Score predictions and return selected indices or corresponding ids."""

    if method == "entropy":
        uncertainty = entropy_uncertainty(probabilities, class_axis=class_axis)
    elif method == "margin":
        uncertainty = margin_uncertainty(probabilities, class_axis=class_axis)
    else:
        raise ValueError("method must be 'entropy' or 'margin'")
    selected = select_active_learning_indices(
        uncertainty, budget, features=embeddings, diversity_weight=diversity_weight, seed=seed
    )
    if frame_ids is None:
        return selected
    if len(frame_ids) != len(uncertainty):
        raise ValueError("frame_ids and probabilities must have the same sample count")
    return [str(frame_ids[index]) for index in selected]


def _sha256_if_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pseudolabel_review_manifest(
    records: Iterable[Mapping[str, Any] | tuple[str | Path, str | Path]],
    *,
    model_checkpoint: str | Path | None = None,
    categories: Mapping[int | str, str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build an auditable, tool-neutral pseudo-label review queue.

    Each item starts in ``pending`` state and has explicit reviewer fields.  A
    record may be a mapping (``image``, ``mask`` and optional confidence data)
    or a simple ``(image, mask)`` tuple.
    """

    items: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = dict(raw) if isinstance(raw, Mapping) else {"image": raw[0], "mask": raw[1]}
        image = record.get("image", record.get("frame"))
        mask = record.get("mask", record.get("pseudolabel"))
        if image is None or mask is None:
            raise ValueError("each review record requires image/frame and mask/pseudolabel")
        item: dict[str, Any] = {
            "id": str(record.get("id", Path(str(image)).stem or index)),
            "image": str(image),
            "mask": str(mask),
            "status": str(record.get("status", "pending")),
            "decision": record.get("decision"),
            "reviewer": record.get("reviewer"),
            "reviewed_at": record.get("reviewed_at"),
            "notes": str(record.get("notes", "")),
        }
        for key in ("confidence", "uncertainty", "selection_rank", "source_index"):
            if key in record and record[key] is not None:
                value = record[key]
                item[key] = float(value) if key in {"confidence", "uncertainty"} else int(value)
        items.append(item)
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("pseudo-label review item ids must be unique")
    return {
        "schema": "sfn.pseudolabel_review",
        "schema_version": 1,
        "seed": int(seed),
        "model": {
            "checkpoint": str(model_checkpoint) if model_checkpoint is not None else None,
            "sha256": _sha256_if_file(model_checkpoint),
        },
        "categories": {str(key): str(value) for key, value in (categories or {}).items()},
        "summary": {"total": len(items), "pending": sum(item["status"] == "pending" for item in items)},
        "items": items,
    }


def save_review_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_pseudolabel_review_manifest(
    records: Iterable[Mapping[str, Any] | tuple[str | Path, str | Path]],
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Build and save a pseudo-label review manifest."""

    return save_review_manifest(build_pseudolabel_review_manifest(records, **kwargs), path)


# Short aliases for notebooks and earlier Phase-14 prototypes.
uncertainty_score = entropy_uncertainty
select_samples = select_active_learning_indices
select_frames = select_active_learning_frames
create_pseudolabel_review_manifest = build_pseudolabel_review_manifest
create_review_manifest = build_pseudolabel_review_manifest
