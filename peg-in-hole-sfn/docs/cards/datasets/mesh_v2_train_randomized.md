# Dataset Card: mesh_v2_train_randomized

> Status: **RECORDED** | Manifest date: 2026-07-13 | Schema: 2

## Identity and provenance

- **Purpose and split:** `train_seen` synthetic dataset.
- **Backend truth label:** `mesh_orthographic`.
- **Manifest:** `data/mesh_v2_train_randomized/manifest.json`.
- **Manifest SHA-256:** `3eb3b4b29f58c39249f4835cd7242d516ad6d8b7bb9f75597979bd594907b506`.
- **Dataset content SHA-256:** `13debc2a4c58e92abec8104baf2db919840152f33692729de23b221f87edabcd`.
- **Seed:** 1300.
- **Generator revision and operator:** not recorded in the manifest.
- **License or usage constraints:** not recorded in the manifest.

The dataset content digest covers the ordered chunk path, sample count, and declared chunk SHA-256 values in canonical JSON form.

## Composition

- **Samples:** 3252 across 7 chunk(s).
- **Shape policy:** `square-triangle`, `square-square`, `square-pentagon`, `square-hexagon`, `square-concave1`, `square-convex1`, `square-convex2`, `square-convex3`, `square-convex4`, `square-fillet1`, `square-fillet2`, `square-fillet3`.
- **Image size:** 250 x 200 pixels.
- **Scale:** 4.0 pixels/mm.
- **Class pixel counts:** background 144007917, peg 10964847, hole 7627236.
- **Pose envelope:** x [-9.999999776482582, 9.999999776482582] mm; y [-9.999999776482582, 9.999999776482582] mm; yaw [-10.0, 10.0] degrees.
- **Edge cases included:** true.
- **Randomization:** medium via sfn.data.augment record version 1.

| Chunk | Samples | Declared SHA-256 | Verified |
|---|---:|---|---|
| `train_seen_000.npz` | 512 | `8424031f9a20bd4832bde92704123c7ab13be708c8fb26189786d094dc463a61` | yes |
| `train_seen_001.npz` | 512 | `52d9519e445b9d9346cd8cae2061e8745de4debd1c23cdcaf982fd08ac81e07b` | yes |
| `train_seen_002.npz` | 512 | `10eb20b72d665267b4d4ec28733b8a28118032627b645aeea05aca0924b43bef` | yes |
| `train_seen_003.npz` | 512 | `3878f815f548620bad9a9bc7567f2c401e7944e5f8f42d4b76018d82acb2d185` | yes |
| `train_seen_004.npz` | 512 | `398909b3083d79a7cd3efd10ce3dacdd5e5aac07cc6f805d034db75ba7c85900` | yes |
| `train_seen_005.npz` | 512 | `d444945b44ec0b96ae15807cb639f709dce19cbf6f61c158383dc7d269981823` | yes |
| `train_seen_006.npz` | 180 | `61cf2658d497cdbdaaeda8b43e367cedea82d1b506dddfd709a193d0506e0c7f` | yes |

## Split integrity

The manifest assigns one split, one seed, and an explicit shape list. Shape overlap and near-duplicate leakage are not assessed by the manifest. The train, validation, and test randomized manifests use distinct seeds and disjoint listed shapes.

## Quality and limitations

All listed chunk hashes match the files present at registry generation time. Labels are simulator-generated mesh-orthographic truth, not sensor measurements or real-robot evidence. Missing values, duplicates, class semantics beyond numeric IDs, and independent annotation review are not reported by the manifest.

## Change record

Recorded from the immutable manifest and verified chunks on 2026-07-13. No reviewer or approval is recorded.
