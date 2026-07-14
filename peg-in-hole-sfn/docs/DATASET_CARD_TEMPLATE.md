# Dataset Card: `<dataset name>`

> Status: **DRAFT** | Owner: `<name>` | Version: `<immutable version>`

## Identity and provenance

- **Purpose:** `<training / validation / test / benchmark>`
- **Artifact URI and checksum:** `<location>` / `<sha256>`
- **Generator code/config revision:** `<git SHA>` / `<config>`
- **Backend truth label:** `<toy_direct / mesh_orthographic / panda_native_camera / real_hardware>`
- **Collection dates and operator:** `<dates>` / `<name or automation>`
- **License/usage constraints:** `<terms>`

## Composition

| Split | Shape policy | Samples / episodes | Seeds | Ground-truth source |
|---|---|---:|---|---|
| `<split>` | `<seen/unseen shape IDs>` | `<count>` | `<seeds>` | `<simulator/sensor/annotation>` |

List schema/version, units, coordinate frames, modalities, resolution, class balance, missing values, duplicates, and edge-case coverage.

## Collection and processing

Describe scene generation or hardware setup, camera calibration, randomization, sampling, filtering, annotation, normalization, and validation. Separate generated truth from measured or inferred labels.

## Split integrity

Explain how shape, episode, seed, and near-duplicate leakage are prevented. State whether test data influenced model or hyperparameter selection.

## Quality and known limitations

Report validation commands/results, corrupt or excluded records, known biases, domain gaps, and unsupported claims. Synthetic or simulated data must not be described as real-robot evidence.

## Reproduction

```text
Environment: <Python/platform/dependency lock>
Collection command: <exact command>
Validation command: <exact command>
Expected manifest/checksum: <value>
```

## Change log

| Date | Version | Change | Reviewer |
|---|---|---|---|
| `<date>` | `<version>` | `<change>` | `<name>` |
