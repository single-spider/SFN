# Dataset Card: mesh_v2_test_randomized

> Status: **RECORDED** | Manifest date: 2026-07-13 | Schema: 2

## Identity and provenance

- **Purpose and split:** `test_unseen` synthetic dataset.
- **Backend truth label:** `mesh_orthographic`.
- **Manifest:** `data/mesh_v2_test_randomized/manifest.json`.
- **Manifest SHA-256:** `e9440b839ad77c8a72636d3d2dc1f30a2847fd8f473201349db34fddff4baf16`.
- **Dataset content SHA-256:** `de55ba6a30e15998887e7c3fbbcdf20c99721eb9566be4afe0964417c245265c`.
- **Seed:** 1302.
- **Generator revision and operator:** not recorded in the manifest.
- **License or usage constraints:** not recorded in the manifest.

The dataset content digest covers the ordered chunk path, sample count, and declared chunk SHA-256 values in canonical JSON form.

## Composition

- **Samples:** 1054 across 3 chunk(s).
- **Shape policy:** `square-concave2`, `square-fillet4`.
- **Image size:** 250 x 200 pixels.
- **Scale:** 4.0 pixels/mm.
- **Class pixel counts:** background 44513410, peg 5150665, hole 3035925.
- **Pose envelope:** x [-9.999999776482582, 9.999999776482582] mm; y [-9.999999776482582, 9.999999776482582] mm; yaw [-10.0, 10.0] degrees.
- **Edge cases included:** true.
- **Randomization:** medium via sfn.data.augment record version 1.

| Chunk | Samples | Declared SHA-256 | Verified |
|---|---:|---|---|
| `test_unseen_000.npz` | 512 | `341ffeb470e940d65df47342a31e1cad14f13d839aba780e0d2e1895413ff64f` | yes |
| `test_unseen_001.npz` | 512 | `e7e390c0df9f156993b9ae1416ec0a9d8a5fb40b8af07cffb9603601703ea133` | yes |
| `test_unseen_002.npz` | 30 | `2afe4d0f502cef432652386552f2dec731887067ea487efb5836fae86162a077` | yes |

## Split integrity

The manifest assigns one split, one seed, and an explicit shape list. Shape overlap and near-duplicate leakage are not assessed by the manifest. The train, validation, and test randomized manifests use distinct seeds and disjoint listed shapes.

## Quality and limitations

All listed chunk hashes match the files present at registry generation time. Labels are simulator-generated mesh-orthographic truth, not sensor measurements or real-robot evidence. Missing values, duplicates, class semantics beyond numeric IDs, and independent annotation review are not reported by the manifest.

## Change record

Recorded from the immutable manifest and verified chunks on 2026-07-13. No reviewer or approval is recorded.
