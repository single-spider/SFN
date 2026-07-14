# Showcase public manifest

`manifest.json` is a compact static-site input generated from the release
artifacts and a corrected dynamic Panda benchmark.  Regenerate it from the
project root after the Panda run completes:

```powershell
..\.venv\Scripts\python.exe scripts\showcase_export.py `
  --panda-root artifacts\showcase_corrected_panda_20260714
```

The exporter carries SHA-256 hashes of every input and fails if the 16-shape
split manifests are inconsistent.  It deliberately excludes the earlier Panda
matrix so the public results never mix simulator configurations.

## Data dictionary

| Field | Meaning |
|---|---|
| `schema_version` | Versioned public manifest identifier. |
| `sources` | Repository-relative input paths and SHA-256 hashes. |
| `shape_splits` | The disjoint train/validation/test assignment of all 16 geometries. |
| `metrics.cartesian` | Historical Cartesian values, not Panda measurements. |
| `metrics.mesh` | Mesh-faithful synthetic benchmark summaries. |
| `panda` | Corrected dynamic PyBullet Panda insertion matrix. |
| `success_rate_wilson_95` | Inclusive Wilson 95% confidence interval. |
| `mean_final_xy_error_mm` | Mean horizontal peg-to-hole-centre error in millimetres. |

All displayed results are simulation results; the manifest makes no hardware
performance claim.
