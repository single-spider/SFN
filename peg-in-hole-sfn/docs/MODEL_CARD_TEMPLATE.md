# Model Card: `<model name>`

> Status: **DRAFT** | Owner: `<name>` | Updated: `<YYYY-MM-DD>`

## Summary

- **Task:** `<segmentation / position / orientation / control>`
- **Artifact:** `<immutable checkpoint URI and checksum>`
- **Code/config revision:** `<git SHA>` / `<config path or embedded config>`
- **Backend truth label:** `<toy_direct / mesh_orthographic / panda_native_camera / real_hardware>`
- **Training dataset:** `<dataset card link and immutable version>`

## Intended use and exclusions

Describe supported inputs, operating envelope, users, and decisions. Explicitly list unsupported uses and whether the model has been tested on unseen meshes, contact insertion, or real hardware.

## Model and training

Record architecture, preprocessing, objective, optimizer, schedule, seeds, compute hardware, wall time, stopping rule, and checkpoint-selection rule.

## Evaluation

| Split / shape policy | Backend truth label | Metric | Result | Run artifact |
|---|---|---:|---:|---|
| `<split>` | `<label>` | `<metric>` | `<value ± uncertainty>` | `<URI>` |

Document baselines, sample counts, confidence intervals, failures, and whether masks/poses are ground truth or predicted.

## Limitations, risks, and ethics

Describe domain gaps, sensitivity to calibration/lighting/shape/clearance, unsafe failure modes, data limitations, and required human or robot safety controls.

## Reproduction

```text
Environment: <Python/platform/dependency lock>
Command: <exact command>
Seed(s): <values>
Outputs: <URI and checksums>
```

## Change log and approval

| Date | Revision | Change | Reviewer |
|---|---|---|---|
| `<date>` | `<SHA>` | `<change>` | `<name>` |
