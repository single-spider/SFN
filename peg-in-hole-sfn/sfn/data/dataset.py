from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np


class NPZDataset:
    """Manifest-aware NPZ dataset.

    Older smoke code only opened the first manifest chunk.  Production-ish
    training needs multiple chunks so large datasets can be generated without
    building one monster compressed file.
    """

    def __init__(self, path: str | Path):
        self.root = Path(path)
        self.manifest = None
        self.chunk_paths: list[Path] = []
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {self.root}. "
                "Collect it first with scripts/collect_dataset.py, or fix --dataset/--val-dataset."
            )
        if self.root.is_dir():
            manifest_path = self.root / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Dataset directory is missing manifest.json: {self.root}. "
                    "This is probably not a collected SFN dataset directory."
                )
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks = self.manifest.get("chunks") or []
            for chunk in chunks:
                name = chunk["path"] if isinstance(chunk, dict) else chunk
                self.chunk_paths.append(self.root / name)
            if not self.chunk_paths:
                raise ValueError(f"Dataset manifest has no chunks: {manifest_path}")
            missing = [p for p in self.chunk_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Dataset manifest references missing chunk(s), first missing: {missing[0]}. "
                    "Regenerate the dataset or restore the missing NPZ chunks."
                )
        else:
            self.chunk_paths = [self.root]

        # Materialize each compressed NPZ member exactly once. Keeping NpzFile
        # handles open causes every random sample access to re-decompress a
        # complete member, makes Windows DataLoader workers unpicklable, and
        # was over an order of magnitude slower during training.
        self._chunks: list[dict[str, np.ndarray]] = []
        for path in self.chunk_paths:
            with np.load(path, allow_pickle=False) as archive:
                self._chunks.append({name: np.asarray(archive[name]) for name in archive.files})
        self._lengths = [int(c["rgb"].shape[0]) for c in self._chunks]
        self._offsets = np.cumsum([0] + self._lengths).tolist()
        self.length = int(self._offsets[-1])
        # Backwards compatibility for old tests/callers that inspect arrays.
        self.arrays = self._chunks[0]
        self.path = self.chunk_paths[0]

    def __len__(self):
        return self.length

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        chunk_idx = bisect_right(self._offsets, index) - 1
        local_idx = index - self._offsets[chunk_idx]
        return chunk_idx, local_idx

    def __getitem__(self, index: int):
        chunk_idx, local_idx = self._locate(int(index))
        arrays = self._chunks[chunk_idx]
        return {k: value[local_idx] for k, value in arrays.items()}

    def close(self) -> None:
        self._chunks.clear()
