from __future__ import annotations

from ..constants import DEFAULT_SHAPE_SPLITS

SPLIT_VERSION = "shape-disjoint-v1"
SPLIT_DEFINITIONS = {SPLIT_VERSION: DEFAULT_SHAPE_SPLITS}

# Deterministic four-fold cross-shape protocol. Each shape is held out exactly
# once; this complements (and does not change) the canonical train/val/test split.
_ALL_SHAPES = sorted({shape for values in DEFAULT_SHAPE_SPLITS.values() for shape in values})
CROSS_SHAPE_FOLD_VERSION = "cross-shape-4fold-v1"
CROSS_SHAPE_FOLDS = {
    f"fold_{fold}": {
        "train": [shape for index, shape in enumerate(_ALL_SHAPES) if index % 4 != fold],
        "test": [shape for index, shape in enumerate(_ALL_SHAPES) if index % 4 == fold],
    }
    for fold in range(4)
}


def get_split(name: str, version: str = SPLIT_VERSION) -> list[str]:
    try:
        definitions = SPLIT_DEFINITIONS[version]
    except KeyError as exc:
        raise KeyError(f"Unknown split version {version!r}; expected {sorted(SPLIT_DEFINITIONS)}") from exc
    try:
        return list(definitions[name])
    except KeyError as exc:
        raise KeyError(f"Unknown split {name!r}; expected {sorted(definitions)}") from exc


def get_cross_shape_fold(fold: int, partition: str = "test") -> list[str]:
    if partition not in {"train", "test"}:
        raise KeyError("Cross-shape fold partition must be 'train' or 'test'")
    try:
        return list(CROSS_SHAPE_FOLDS[f"fold_{int(fold)}"][partition])
    except (KeyError, ValueError) as exc:
        raise KeyError("Cross-shape fold must be one of 0, 1, 2, 3") from exc


def validate_shape_disjointness(splits: dict[str, list[str]] | None = None) -> None:
    splits = splits or DEFAULT_SHAPE_SPLITS
    seen = {}
    for split, shapes in splits.items():
        for shape in shapes:
            if shape in seen:
                raise ValueError(f"Shape {shape!r} appears in both {seen[shape]!r} and {split!r}")
            seen[shape] = split
