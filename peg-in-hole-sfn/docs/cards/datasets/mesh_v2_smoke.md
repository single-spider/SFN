# Dataset Card: mesh_v2_smoke

> Status: **RECORDED** | Manifest date: 2026-07-13 | Schema: 1

## Identity and provenance

- **Purpose and split:** `test_unseen` synthetic dataset.
- **Backend truth label:** `mesh_orthographic`.
- **Manifest:** `data/mesh_v2_smoke/manifest.json`.
- **Manifest SHA-256:** `b78a3d1d7dc29e4855e8d019a3c48e2d8b5e033468cdc08a392ae8dd4335789e`.
- **Dataset content SHA-256:** `a210c1348195e12afc4053562c962e0b4f73e2e0bcad3c3864e37411dc2320e1`.
- **Seed:** 7300.
- **Generator revision and operator:** not recorded in the manifest.
- **License or usage constraints:** not recorded in the manifest.

The dataset content digest covers the ordered chunk path, sample count, and declared chunk SHA-256 values in canonical JSON form.

## Composition

- **Samples:** 34 across 1 chunk(s).
- **Shape policy:** `square-concave2`, `square-fillet4`.
- **Image size:** 250 x 200 pixels.
- **Scale:** 4.0 pixels/mm.
- **Class pixel counts:** background 1444766, peg 164890, hole 90344.
- **Pose envelope:** x [-9.999999776482582, 9.999999776482582] mm; y [-9.999999776482582, 9.999999776482582] mm; yaw [-10.0, 10.0] degrees.
- **Edge cases included:** true.
- **Randomization:** not recorded.

| Chunk | Samples | Declared SHA-256 | Verified |
|---|---:|---|---|
| `test_unseen_000.npz` | 34 | `c3b7e4c3a610f6ad9b6072001774b21d9a77b5e88a6198efb946c7ad359f591e` | yes |

## Split integrity

The manifest assigns one split, one seed, and an explicit shape list. Shape overlap and near-duplicate leakage are not assessed by the manifest. The train, validation, and test randomized manifests use distinct seeds and disjoint listed shapes.

## Quality and limitations

All listed chunk hashes match the files present at registry generation time. Labels are simulator-generated mesh-orthographic truth, not sensor measurements or real-robot evidence. Missing values, duplicates, class semantics beyond numeric IDs, and independent annotation review are not reported by the manifest.

## Change record

Recorded from the immutable manifest and verified chunks on 2026-07-13. No reviewer or approval is recorded.
