# PEC-cylinder qualification matrix — 22 September 2026

The restartable `scripts/qualify_pec_cylinders.py` campaign completed 31 cases.
Five off-grid PEC cylinders span radii 0.045–0.12 m, different subcell positions,
and incidence angles from 0 to 3.91 radians. Enlarged conformal grids use
48, 64, 96, and 128 cells per axis. Each scene also has a plain-conformal 64²
comparison. The medium cylinder has independent duration, PML, and near-to-far
contour variations at 96² and 128².

Every case stores complex128 numerical and analytic far fields at 0.8, 1.0, and
1.2 GHz over 180 observation angles. Per-case JSON and NPZ files are written
atomically. Resume checks the complete physics-source hash, configuration hash,
array keys, shapes, finiteness, frequency grid, and angle grid before reusing a case.
A second full invocation reused all 31 verified cases.

## Results

| Check | Measured | Proposed gate | Result |
|---|---:|---:|---|
| Worst 128² complex angular L2 | 0.9509% | <1% | pass |
| Worst 128² weighted phase RMS | 0.5090° | <1° | pass |
| Worst 128² last-window field tail | 6.33e-7 | <1e-5 | pass |
| Analytic series-order change | 0 | <1e-10 | pass |
| Worst 128² contour-induced complex change | 0.5209% | <0.5% | **fail** |

All five 128² enlarged cases recovered the background-grid time step. At 64²,
enlargement used 2.11–8.22 times fewer cell updates than plain conformal and had
2.57–3.33 times lower joint loss for every scene. This is favorable finite-matrix
evidence, not a general stability or accuracy proof.

Duration and PML changes alter the 128² complex field by at most 9.4e-7 and 2.1e-6,
respectively. Enlarging the near-to-far contour changes it by 0.224%, 0.397%, and
0.521% across the three frequencies. The last value narrowly misses the proposed
0.5% observation gate, so M1 is not complete. The next numerical action is monitor
spatial-convergence work, followed by close-gap/cavity and mixed-topology tests.

Local artifacts are under `runs/pec_cylinder_qualification/`, including
`report.json`, the per-case files, and `qualification.png`. They are intentionally
ignored by Git; this document records the reproducible measured result.
