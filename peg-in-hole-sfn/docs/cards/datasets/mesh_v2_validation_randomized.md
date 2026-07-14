# Dataset Card: mesh_v2_validation_randomized

> Status: **RECORDED** | Manifest date: 2026-07-13 | Schema: 2

## Identity and provenance

- **Purpose and split:** `validation_unseen` synthetic dataset.
- **Backend truth label:** `mesh_orthographic`.
- **Manifest:** `data/mesh_v2_validation_randomized/manifest.json`.
- **Manifest SHA-256:** `8a5348409c16bee713a8c5d485535404913c591a46a337e91f9712949b68ed6b`.
- **Dataset content SHA-256:** `3324c44e6414954ff09a4638d587e638712eb0be95b5ddfc3ca31ae4de18c6c7`.
- **Seed:** 1301.
- **Generator revision and operator:** not recorded in the manifest.
- **License or usage constraints:** not recorded in the manifest.

The dataset content digest covers the ordered chunk path, sample count, and declared chunk SHA-256 values in canonical JSON form.

## Composition

- **Samples:** 1054 across 3 chunk(s).
- **Shape policy:** `square-diamond`, `square-trapezoid`.
- **Image size:** 250 x 200 pixels.
- **Scale:** 4.0 pixels/mm.
- **Class pixel counts:** background 47700952, peg 2875589, hole 2123459.
- **Pose envelope:** x [-9.999999776482582, 9.999999776482582] mm; y [-9.999999776482582, 9.999999776482582] mm; yaw [-10.0, 10.0] degrees.
- **Edge cases included:** true.
- **Randomization:** medium via sfn.data.augment record version 1.

| Chunk | Samples | Declared SHA-256 | Verified |
|---|---:|---|---|
| `validation_unseen_000.npz` | 512 | `c3a7997003f1e4618bf3d7e0432ad1368d10cb706fe8f9820b5871a7a3e328b2` | yes |
| `validation_unseen_001.npz` | 512 | `ec7b0d86b6f74e464b016f6d4f074bb8a6180f49658fd6b4e210f44a7f5faf0b` | yes |
| `validation_unseen_002.npz` | 30 | `a7a4e821bc11fd0fb9e2e3ad6fee61f7fd6a6cc0b98fc75125835b4aaba6468e` | yes |

## Split integrity

The manifest assigns one split, one seed, and an explicit shape list. Shape overlap and near-duplicate leakage are not assessed by the manifest. The train, validation, and test randomized manifests use distinct seeds and disjoint listed shapes.

## Quality and limitations

All listed chunk hashes match the files present at registry generation time. Labels are simulator-generated mesh-orthographic truth, not sensor measurements or real-robot evidence. Missing values, duplicates, class semantics beyond numeric IDs, and independent annotation review are not reported by the manifest.

## Change record

Recorded from the immutable manifest and verified chunks on 2026-07-13. No reviewer or approval is recorded.
