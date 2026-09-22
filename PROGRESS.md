# Scattering project progress — 22 September 2026

## Archive and new project

- Previous source: `codex/archive-receiver-cnn-20260922`, commit `5878ddd`.
- Active branch: `codex/plane-wave-scattering`.
- Old project is preserved under `archive/receiver_cnn_2026-09-22/`: 95,496 files,
  1,539,867,582 bytes, and 404 resolving symlinks in the archive inventory.
- Root `artifacts` is a compatibility link to the old artifact tree. No new dataset
  or training campaign was launched. Ignored datasets remain local, outside Git.

## Implemented

The independent `scattermesh` package now includes:

- CPU float64 nonuniform TMz Yee updates, oblique analytic incident plane waves,
  sampled dielectric averaging, electric conductivity, and scattered-field CPML.
- Streaming complex surface DFT with correct E/H time staggering and a closed
  2D near-to-far transform. Outputs preserve source-normalized complex fields
  in sqrt(m) and derived scattering width in metres.
- Exact circle/rectangle PEC cut edges without mesh anchors, a staircase baseline,
  and experimental cell enlargement using consistent mass/curl projection.
- Cut-aware and enlarged-operator stability bounds, source-spectrum checks,
  unsupported-geometry rejection, and runtime/Nt/field-tail diagnostics.
- [openEMS implementation review](docs/openems_review.md) and the updated
  [implementation plan](IMPLEMENTATION_PLAN.md).

## Verification

`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q`: **26 passed**.
Ruff checks and formatting pass. Tests include an analytic outgoing Hankel field
for NF2FF phase/normalization; lossless/lossy cylinders; nonempty dielectric ratio-2.8
grading; PEC scattering; roundoff versus physical cuts; projected mass/stiffness
and eigenvalues; source-active total-field constraints; a 1e-6 cut fraction; absolute complex
PEC-cylinder phase; and phase-only/translation regression checks.

### Off-grid PEC and cell enlargement

The pilot completed **35 simulations**, including **17 enlarged runs**. All
enlarged runs recovered the background-grid dt (within floating-point roundoff),
with the same 0.9 CFL safety factor. All runs returned finite fields; the largest
last-10%-window scattered-E tail/peak was 4.74e-6. These are measured finite-run
results, not a universal late-time stability theorem.

Example: displaced PEC cylinder, radius 0.08 m, center (0.613, 0.591) m, incidence
0.7 radians, domain 1.2 m square. Complex-field and width errors are angular L2
errors against the analytic solution at 1 GHz. Phase RMS is reference-power weighted
over non-null angles. CPU elapsed time includes setup and observations.

| Mesh | Treatment | Complex error | Width error | Phase RMS | Nt | Time |
|---|---|---:|---:|---:|---:|---:|
| Uniform 64² | Staircase | 21.99% | 14.43% | 12.35° | 880 | 0.36 s |
| Uniform 64² | Conformal | 2.83% | 1.60% | 1.55° | 1,861 | 0.79 s |
| Uniform 64² | Enlarged | 1.69% | 1.77% | 0.83° | 880 | 0.45 s |
| Nonuniform 64² | Staircase | 8.74% | 6.54% | 4.69° | 1,946 | 1.14 s |
| Nonuniform 64² | Conformal | 3.14% | 2.23% | 1.62° | 8,639 | 5.00 s |
| Nonuniform 64² | Enlarged | 2.89% | 2.45% | 1.41° | 1,946 | 1.21 s |
| Nonuniform 96² | Conformal | 1.41% | 0.89% | 0.75° | 17,499 | 12.78 s |
| Nonuniform 96² | Enlarged | 1.30% | 0.92% | 0.68° | 2,920 | 2.76 s |
| Uniform 128² | Conformal | 0.71% | 0.35% | 0.39° | 4,801 | 4.75 s |
| Uniform 128² | Enlarged | 0.44% | 0.40% | 0.22° | 1,759 | 2.28 s |

The enlarged uniform 128² width errors were 0.29%, 0.40%, and 0.43% at 0.8, 1.0, and
1.2 GHz. The method preserves PEC intersections and avoids anchoring, with a modest
accuracy cost in these examples. The focused nonuniform mesh is not always better
than uniform: its coarser exterior and extra Nt matter. It is a solver test grid,
not an optimized teacher or evidence of CNN headroom by itself.

Rectangle small-cell stress: at cut fraction 0.002, plain conformal needed 12,679
steps and 4.86 s; enlarged mode needed 1,131 steps and 0.52 s for the same 60 ns
simulation. A 1e-6 fraction also ran with 1,131 steps, full grid dt, finite fields,
and tail/peak 1.96e-6. Rectangle cases establish stability/settling here, not
independent analytic scattering accuracy.

Results: `runs/conformal_pec_pilot/report.json`, `spectra.npz`, `conformal_pec.png`.
Complex128 far fields and incident spectra are saved in `spectra.npz`;
`complex_far_field.png` plots real/imaginary components and phase error. A tracked
copy of the numeric report is in `docs/validation/conformal_pec.json`.
See [method details and limitations](docs/conformal_pec.md).

### Dielectric solver foundation

The separate 12-run dielectric check compares uniform/focused meshes at 64²/128²,
two incidence angles, three frequencies, and four sensitivity changes. For the
128² nonuniform cylinder at 30° incidence, angular width errors were 0.23%, 0.35%,
and 0.61% at 0.8, 1.0, and 1.2 GHz. **Complex-field errors were 0.50%,
0.89%, and 1.59%; weighted phase RMS was 0.26°, 0.48°, and 0.88°.**
Width sensitivity maxima across frequencies:

| Change | Relative angular width change |
|---|---:|
| Extend simulation from 30 to 40 ns | 0.0000024% |
| PML thickness 0.15 to 0.20 m | 0.00069% |
| Enlarge NF2FF contour | 0.679% |
| Material quadrature 12² to 24² samples | 0.0994% |

Complex-field contour sensitivity reaches 0.975%, exceeding the proposed 0.5%
qualification gate. The 1.2 GHz complex-field error also exceeds 1%. Production
reference acceptance is deliberately not implemented yet. Results are under
`runs/scattering_bootstrap/`; a tracked report is in
`docs/validation/dielectric_bootstrap.json`.

## Joint loss and initial training curriculum

The referenced discussion was read and incorporated in
[the curriculum](docs/curriculum.md) and the implementation plan. Start with single
PEC cylinders, then dielectric cylinders, using analytic complex references.
Introduce multiple cylinders and gaps before rectangles/corners and complex scenes.
The proposed first pool is 32–64 base geometries for candidate-mesh experiments;
no dataset worker or CNN training has been launched.

`scattermesh.metrics.scattering_loss` now scores both normalized complex MSE and
floored log-RCS error. Initial weights are 1 and 0.25 after explicit dB scaling,
with a -40 dB relative RCS floor; these are provisional pilot choices. Both
components are saved for calibration. Phase-only and null-field tests check the
loss, and all 38 saved analytic-cylinder cases are scored without rerunning FDTD.
Reports retain the field-run code hashes and separate loss-postprocessing provenance.
Rectangle stability cases have no invented analytic accuracy scores.

Compare candidates at matched cell-update budgets, retain multiple good meshes,
and group related geometry variants into the same split. Demonstrate useful
low-budget mesh improvements before fitting a new small CNN.

## Next work

1. Broaden PEC enlargement validation to close gaps, cavities, resonances, strong
   grading, mixed dielectrics, and split-edge thin screens. Compare accuracy/cost
   before choosing a default. Plain conformal remains the current default.
2. Resolve observation/CPML convergence across the benchmark range; add the
   openEMS-style TFSF comparison and an external solver cross-check.
3. Port qualified updates, PEC aggregation transfers, surface DFT, and NF2FF to GPU.
4. Run the small single-cylinder candidate search with the joint loss and matched
   update budgets, then train only if there is useful mesh headroom. Expand the
   curriculum gradually and report simple/sparse/complex families separately.
