"""Shape asset discovery and validation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ShapeAssets:
    shape: str
    root: Path
    base_urdf: Path
    base_obj: Path
    peg_obj: Path
    peg_test_urdf: Path
    mask_obj: Path
    peg_urdf: Path | None = None


@dataclass
class AssetValidationResult:
    shape: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg):
        self.errors.append(str(msg))
        self.valid = False

    def add_warning(self, msg):
        self.warnings.append(str(msg))


class AssetRegistry:
    def __init__(self, root: str | Path | None = None):
        self.root = (
            Path(root).resolve()
            if root is not None
            else (Path(__file__).resolve().parents[2] / "gymEnv" / "envs" / "complex").resolve()
        )

    def list_shapes(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and p.name != "franka_panda")

    def get(self, shape: str) -> ShapeAssets:
        if shape not in self.list_shapes():
            raise KeyError(f"Unknown shape {shape!r}; no fallback asset will be used")
        root = (self.root / shape).resolve()
        peg_urdf = root / "peg" / "peg.urdf"
        return ShapeAssets(
            shape,
            root,
            (root / "base" / "base.urdf").resolve(),
            (root / "base" / "base.obj").resolve(),
            (root / "peg" / "peg.obj").resolve(),
            (root / "peg" / "peg_test.urdf").resolve(),
            (root / "mask.obj").resolve(),
            peg_urdf.resolve() if peg_urdf.exists() else None,
        )

    def validate(self, shape: str, strict_dependencies: bool = False) -> AssetValidationResult:
        r = AssetValidationResult(shape, True)
        try:
            a = self.get(shape)
        except Exception as exc:
            r.add_error(exc)
            return r
        for label, path in {
            "base/base.urdf": a.base_urdf,
            "base/base.obj": a.base_obj,
            "peg/peg.obj": a.peg_obj,
            "peg/peg_test.urdf": a.peg_test_urdf,
            "mask.obj": a.mask_obj,
        }.items():
            if not path.exists():
                r.add_error(f"Missing required file {label}: {path}")
            elif path.stat().st_size <= 0:
                r.add_error(f"Required file is empty {label}: {path}")
        if r.errors:
            return r
        self._validate_meshes(a, r, strict_dependencies)
        self._validate_urdfs(a, r, strict_dependencies)
        return r

    def validate_all(
        self, shapes: Iterable[str] | None = None, strict_dependencies: bool = False
    ) -> dict[str, AssetValidationResult]:
        return {
            s: self.validate(s, strict_dependencies)
            for s in (list(shapes) if shapes is not None else self.list_shapes())
        }

    def _validate_meshes(self, a, r, strict):
        try:
            import trimesh
        except ModuleNotFoundError:
            (r.add_error if strict else r.add_warning)("trimesh not installed; mesh finite-bounds checks skipped")
            return
        for label, path in [("base.obj", a.base_obj), ("peg.obj", a.peg_obj), ("mask.obj", a.mask_obj)]:
            try:
                mesh = trimesh.load_mesh(str(path), force="mesh")
                bounds = getattr(mesh, "bounds", None)
                if bounds is None or not bool(bounds.size):
                    r.add_error(f"{label} has no mesh bounds")
                    continue
                if not all(math.isfinite(float(v)) for row in bounds for v in row):
                    r.add_error(f"{label} has non-finite mesh bounds")
                if float(max(mesh.extents)) <= 0:
                    r.add_error(f"{label} has empty extents")
            except Exception as exc:
                r.add_error(f"Could not load {label} with trimesh: {exc}")

    def _validate_urdfs(self, a, r, strict):
        try:
            import pybullet as p
        except ModuleNotFoundError:
            (r.add_error if strict else r.add_warning)("pybullet not installed; URDF load/collision checks skipped")
            return
        cid = p.connect(p.DIRECT)
        try:
            for label, path in [("base.urdf", a.base_urdf), ("peg_test.urdf", a.peg_test_urdf)]:
                try:
                    body = p.loadURDF(str(path), useFixedBase=True, physicsClientId=cid)
                    if label == "peg_test.urdf":
                        if not p.getVisualShapeData(body, physicsClientId=cid):
                            r.add_error("standalone peg has no visual geometry")
                        if not p.getCollisionShapeData(body, -1, physicsClientId=cid):
                            r.add_error("standalone peg has no collision geometry")
                except Exception as exc:
                    r.add_error(f"Could not load {label} in PyBullet DIRECT: {exc}")
        finally:
            p.disconnect(cid)
