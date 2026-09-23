# Full simple-dielectric candidate campaign

Campaign `simple_candidate_full_d535ccac015b3cd8` completed all 3,168 planned
cases for dataset `simple_dk_3ea40e8117434e5c`. The cache contains 3,146 accepted
simulations and 22 explicitly unsettled candidates, with no missing or malformed
records. All uniform baselines settled, so the report accepted all 88 illumination
groups and produced 352 budget labels.

The frozen pre-M5 audit does **not** authorize CNN training. Training contains ten
meaningful matched-update wins spread across three lineages, two candidate policies,
and the 64- and 96-cell budgets. The held-out epsilon_r=5 test lineage contains
eleven wins, with a maximum 2.003x improvement. The epsilon_r=3.25 validation
lineage contains no nonuniform win: uniform is selected for all 32 geometry,
illumination, and budget labels. This fails only the predeclared
`validation_has_headroom` check.

| Split | Illumination groups | Budget labels | >=1.05x wins | Winning policies | Winning budgets |
|---|---:|---:|---:|---|---|
| Train | 72 | 288 | 10 | `region_medium`, `region_strong` | 64, 96 |
| Validation | 8 | 32 | 0 | none | none |
| Test | 8 | 32 | 11 | `region_strong` | 64, 96 |

The validation result is not a threshold near miss. Under the available update
caps, its best nonuniform alternative reaches at most 0.940 times the uniform
accuracy ratio; most alternatives are substantially worse. However, equal-cell
comparisons show that focused meshes often lower error while raising time steps.
The campaign sampled candidate resolutions only at 32, 48, 64, and 96 cells. At a
64-cell uniform update cap, for example, a same-resolution focused candidate is
often more accurate but unaffordable, while the next available focused candidate
has only 48 cells. This coarse candidate-resolution ladder can hide the useful
matched-cost point between them.

The next search iteration therefore keeps the gate and held-out evidence intact,
but gives each nonuniform policy several candidate-specific resolution factors.
Those factors are selected on training lineages first, then frozen before another
validation/test evaluation. The scene pool will also be revised before fitting a
model because its original lineages correlate radius, permittivity, and
conductivity. A factorial or space-filling pool must vary these inputs
independently so the CNN cannot use object size as a proxy for material contrast.

The authoritative machine-readable outputs are
`runs/simple_candidate_pilot/report.json` and
`runs/simple_candidate_pilot/labels.json`.
