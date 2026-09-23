# Mesh-CNN v2 progress — 23 September 2026

- Archived previous scattering/CNN source and 59,178 historical run files with
  SHA-256 inventory; retained a local `runs/` compatibility link. Preserved the
  receiver-CNN archive and repaired/verified all 404 relocated symlinks. The
  active environment and editable install resolve in this project.
- Added continuous ellipse, polygon and smooth-lobed geometry; C0–C2 lineage
  generation; candidate, scoring, reference, dataset, large/small model,
  training, evaluation and deadline-aware campaign components.
- The 16-scene sizing gate ran on four TITAN RTX GPUs in 109 seconds. Only 9/16
  rows had all projected components usable. The compatible rows alone imply a
  **partial projection of 18.16 elapsed hours** for 128 lineages and 36.31
  hours for 256 lineages, using all four GPUs. The 14-hour data-phase gate
  therefore stopped bulk execution before training or frozen testing.
- A complete dielectric-circle reference smoke escalated to 512 cells per axis
  and passed duration, quadrature, contour, and PML probes; its maximum
  measured reference variation was 0.1853%. New star dielectric and
  smooth-lobed PEC CPU/CUDA complex fields agreed within 5e-16 relative L2.
- Imported 1,104 accepted historical training examples into a portable local
  dataset, preserving source checksums and rescoring at exponent 0.05. No
  historical validation or test examples entered the import.
- Details and remediation conditions are in [the v2 pilot record](docs/v2_pilot.md).

The remainder of this file records the archived scattering project as it stood
on 22 September 2026. The exact original is in the scattering source archive.

---

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

`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q`: **28 passed, 1 CUDA-visibility skip**.
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

### Expanded PEC-cylinder qualification

The restartable 35-case single-cylinder matrix now spans five sizes/positions/
incidence conditions, 48²–128² enlarged meshes, plain-conformal 64² comparisons,
and independent duration/PML/contour variations. A second invocation verified and
reused all per-case caches using source/configuration fingerprints and complex-array
checks. At 128², worst complex angular L2 was 0.951%, worst phase RMS was 0.509°,
and all enlarged cases retained the full background time step.

At 64², enlargement needed 2.11–8.22 times fewer updates and produced 2.57–3.33
times lower joint loss than plain conformal across all five scenes. Duration and PML
sensitivity are negligible in the tested baseline. Near-to-far contour sensitivity
at 1.2 GHz is 0.521% at 128², but falls to 0.334% at 160². Reference generation
must therefore check this independently and escalate when needed. Broader M1
qualification remains open for PEC close gaps, cavities, thin screens, contacting
PEC/dielectric combinations, and external comparison. See the
[qualification record](docs/validation/pec_cylinder_qualification.md) and local
`runs/pec_cylinder_qualification/qualification.png`.

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

The next restartable 25-case dielectric matrix separates the curriculum by measured
difficulty. At 192², the initial epsilon_r<=4 family passes: worst complex error
1.403%, phase RMS 0.671°, tail 2.67e-8, and worst duration/PML/contour/quadrature
variation 0.232%. These cases have at least 19.99 cells per shortest internal
wavelength.

At 128², epsilon_r=12 and 30 initially had only 7.69 and 4.87 cells per shortest
internal wavelength, with large error and resonant tails. CUDA escalation now accepts
the lossless epsilon_r=12 cylinder at 512²/400 ns: maximum complex error 1.668%,
phase RMS 0.842°, and tail/peak 7.20e-6. Duration, quadrature, and contour changes
are at most 0.0358%. A conductive epsilon_r=30, sigma_e=0.2 S/m cylinder passes at
512²/50 ns with 0.885% complex error and at most 0.0340% independent variation.

The lossless epsilon_r=30 high-Q scene remains nonconverged and is skipped. At the
bounded pilot hard limit, 512²/400 ns still has tail/peak 0.0356 and a 1.979%
duration change; a 768²/100 ns probe changes the complex field by 1.896%. See the
[initial dielectric qualification](docs/validation/dielectric_cylinder_qualification.md)
and [high-contrast escalation](docs/validation/dielectric_reference_escalation.md).

### CUDA dielectric backend

The M2 backend now runs dielectric and PEC-only nonuniform Yee/CPML updates,
analytic plane-wave forcing, and stagger-aware streaming complex DFTs on a Torch device.
Float64 CUDA agrees with NumPy final fields and far fields to below 5e-15 in the
bounded real-GPU test. At 512² and 2,010 steps, one TITAN RTX completes the benchmark
in 4.79 s versus 34.65 s for NumPy, a 7.23x speedup. At 128² CUDA startup/launch
overhead still makes it slower. See the [CUDA validation](docs/validation/cuda_foundation.md).

Staircase, conformal, and enlarged PEC modes now match NumPy across fields, DFTs,
far fields, CFL diagnostics, and projected transfers. On the TITAN RTX, enlarged
PEC at 512²/2,010 steps takes 7.03 s versus 37.03 s for NumPy, a 5.27x speedup;
complex far fields differ by 4.13e-15. The restartable per-attempt runner has
exercised all four GPUs independently and retains failed attempts. Float32 passes
complex-field, phase, analytic-degradation, and settling gates for accepted
epsilon_r=12, conductive epsilon_r=30, and enlarged-PEC cases. It is slower than
float64 in both 512² probes (0.87x dielectric and 0.78x PEC), so float64 remains
the production default. See the [precision record](docs/validation/cuda_precision.md).
The restartable campaign scheduler now pins tasks across four GPUs, validates current
solver hashes before cache reuse, writes atomic state and per-attempt logs, retains
scientific nonconvergence, and retries only process failures. A real four-GPU smoke
completed four concurrent attempts in one launch and resumed without another GPU
run. See the [scheduler validation](docs/validation/multi_gpu_scheduler.md).
The adaptive policy layer now interprets individual accuracy/phase/settling gates,
promotes mesh and time independently, requires spatial/duration/quadrature/contour
agreement, and records hard-limit skips. Its conductive epsilon_r=30 pilot accepted
the 512²/50 ns base after five attempts in two waves; the largest independent change
was 0.3374% under 640² spatial refinement. Resume revalidated all artifacts without
GPU work. See the [policy validation](docs/validation/adaptive_convergence_policy.md).
Separated mixed PEC/dielectric coupling now passes CPU/GPU equivalence and three
complex-field self-convergence profiles: wide-gap circles, a 15 mm close gap, and a
PEC rectangle with lossy epsilon_r=12 material. Maximum spatial changes are 0.123%,
0.433%, and 0.392%; all duration/quadrature/contour changes are below 0.322%, and
all tails are below `1e-5`. See the [mixed validation](docs/validation/mixed_scattering.md).
Contacting/overlapping interfaces and broader scene generation remain open.

## Joint loss and initial training curriculum

The referenced discussion was read and incorporated in
[the curriculum](docs/curriculum.md) and the implementation plan. Start with single
PEC cylinders, then dielectric cylinders, using analytic complex references.
Introduce multiple cylinders and gaps before rectangles/corners and complex scenes.
The initial pool proposal was 32–64 base geometries for candidate-mesh experiments;
the later factorial dielectric pool, candidate campaign, distillation dataset, and
first full CNN fit are recorded below.

The final production target is now explicitly sparse: localized single- and
multi-object dielectric/PEC clusters, including close gaps, embedded in a much
larger domain. Track occupied area and projected x/y feature support because a
tensor grid loses its budget advantage when objects span the whole domain. Dense
assemblies remain capped stress tests; sparse scenes receive at least 70% of later
combined-training sampling weight.

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

The first M3 headroom search evaluated 96 true-uniform and interface-focused PEC-
cylinder candidates at 32/48/64/96 cells per axis with a common 50 ns duration.
Seventy-seven settled, fourteen were geometrically infeasible, and five aggressive
32-cell meshes missed the settling gate. Uniform grids form the complete Pareto
frontier in all four scenes. Accepted focused candidates cost 1.39--3.61 times more
updates and have 1.25--5.44 times more joint complex/RCS loss than same-cell uniform
grids. This teacher family is rejected; training remains gated on a broader object-
region, hybrid, and randomized density search. See the
[headroom pilot](docs/validation/mesh_headroom_pilot.md).

That broader search is now measured. It contains 208 candidates and 96 dense
uniform controls through 128 cells per axis. Mild object-region focus produced an
apparent 2.47% advantage only while controls were four cells apart. A targeted 67²
uniform grid then beat the candidate in both loss and updates. Every final Pareto
point is uniform across all four isolated PEC scenes. M3 therefore moves to simple
dielectric cylinders; PEC cylinders remain valuable analytic validation cases but
do not supply useful mesh-policy labels by themselves.

The follow-up dielectric search passes the headroom gate. Across 208 candidates and
156 dense uniform controls, 355 of 364 runs settle. For epsilon_r=4, a 64² region-
focused grid has 1.896 times lower joint loss than the best affordable 75² uniform
grid at nearly equal updates; a 96² hybrid has a 1.863x advantage over 108² uniform.
The lossy epsilon_r=4 scene reaches 1.448x. Both complex-field and log-width terms
improve, and the best candidate uses only 1.058 grading. See the
[dielectric headroom qualification](docs/validation/dielectric_mesh_headroom.md).
M3 now has enough evidence to begin the larger simple-scene candidate-label pool.

The first M4 pool manifest is now deterministic and tracked as
`simple_dk_3ea40e8117434e5c`: 32 geometries, eight grouped lineages, a 24/4/4
train/validation/test geometry split, disjoint held-out angle sets, and 352 expanded
illumination/budget conditions. Records include physical feature size, its 256²
input-grid span, minimum internal wavelength, material, and analytic-reference type.
The next executable stage is the restartable candidate-label runner for this manifest.

That runner now passes a 270-case end-to-end pilot. It accepts 266 cases, retains
four unsettled randomized candidates, and writes Pareto labels for all ten grouped
illumination sets. Two held-out epsilon_r=5 test conditions show 2.003x and 1.648x
matched-cap improvements. The selected train/validation subset has no positive win,
so training remains blocked by data adequacy rather than software. The generated
full plan `simple_candidate_full_d535ccac015b3cd8` expands all 32 geometries and 352
conditions into 3,168 restartable cases, including the epsilon_r=4.5 train lineage.
See the [candidate pilot](docs/validation/simple_candidate_pilot.md).

The full 3,168-case campaign is complete. It produced 3,146 accepted and 22
unsettled records, no malformed records, complete uniform baselines, and 352 labels
across 88 illumination groups. Training has ten >=1.05x wins across three lineages,
two policies, and two budgets; test has eleven wins and reaches 2.003x. Validation
has no nonuniform win, so the frozen gate correctly returns `not_ready_for_m5`.
See the [full campaign report](docs/validation/simple_candidate_full.md).

The next candidate-search iteration addresses a resolution-sampling gap exposed by
the complete result. Focused grids frequently beat uniform at equal axis counts but
need more time steps, while the existing 32/48/64/96 ladder offers no slightly
smaller candidate that fits the uniform update cap. Add candidate-specific
resolution factors, select them using training lineages, and freeze them before
reevaluating held-out data. The following pool version must also decorrelate radius,
permittivity, and conductivity before CNN fitting.

The decorrelated successor manifest is now generated as
`simple_factorial_69168792914bfe03`: 148 geometries in 74 grouped lineages and
1,552 conditions. Its low/moderate training split is a full 4x3x2
material/size/loss product plus low-contrast lossless controls. A 108-case
qualification sweep then tested epsilon_r={10,20,30} at loss tangent
{0.03,0.10,0.20}. Every case at 0.10 and 0.20 settled in 70 ns across two sizes,
two budgets, and three mesh policies; only 3/36 at 0.03 settled. High-contrast
training now reaches epsilon_r=30 with loss tangent >=0.10, while validation and
test use disjoint levels. See the
[high-permittivity qualification](docs/validation/high_epsilon_loss_qualification.md).

The intermediate-resolution search is complete. Its training-only campaign has
59/72 meaningful wins, and a separately frozen held-out campaign recovers 19/32
validation and 28/32 test wins. This proves the original strict-update failure came
from the coarse resolution ladder. The user has fixed the product contract at exact
requested `Nx` and `Ny`, so scaled candidates are diagnostic only. The next pilot
uses six exact-axis policies and ranks them with a configurable soft `Nt` exponent,
initially 0.1. See the
[scaled candidate validation](docs/validation/scaled_candidate_search.md).

The replacement exact-budget pilot is complete on the expanded high-contrast
pool. All 312 cases settled at 70 ns. It selects nonuniform meshes for 49/52 labels
and passes every readiness check: training has 32/36 meaningful wins, while
validation and test each have 8/8. The full 9,312-case simple-curriculum campaign
uses the same six policies, exact axis budgets, soft Nt exponent 0.1, and adaptive
70/140/560 ns retries. See the
[exact-budget pilot report](docs/validation/factorial_exact_budget_pilot.md).

M5 implementation has started without waiting for the full campaign to finish.
The active residual U-Net predicts a 2D importance map and projects it to positive
x/y density profiles. Inputs explicitly include material channels, signed distance,
interface proximity, feature size, frequency band, incidence direction, and exact
Nx/Ny budget. The dataset builder retains all candidate profiles and physics scores,
rejects partial campaigns, and verifies saved rankings and exact axis counts. The
deterministic inference projection preserves exact Nx/Ny and repairs grading only
when needed. See the [mesh distillation contract](docs/mesh_distillation.md).

The 52-label M5 implementation pilot is also complete. Its 532,593-parameter model
reaches 8.939e-5 validation and 1.036e-4 test profile loss, respectively 20.40x and
18.11x below a uniform-profile baseline. All held-out axes meet the grading cap
without repair. These results qualify the checkpoint for the 16-case learned-mesh
CUDA physics pilot; they do not yet establish scattering improvement. See the
[distillation pilot report](docs/validation/mesh_distillation_pilot.md).

The learned-mesh CUDA physics pilot is now complete. All 16 validation/test meshes
settled at 70 ns. Validation records 7/8 meaningful wins over uniform with 3.135x
median improvement, and test records 8/8 with 2.641x median improvement. Median
soft-Nt score gaps to the searched teacher are 1.356x and 1.259x. One validation
48x48 case is worse than uniform and remains an explicit full-training regression
target. The diagnostic physics gate passes. See the
[learned-mesh physics report](docs/validation/learned_mesh_physics_pilot.md).

The remaining simple-family stages now have executable handoffs. The first workflow
waits for all 9,312 candidate cases to reach a terminal duration attempt, requires
an accepted uniform baseline for each condition, masks hard-limit rejected
nonuniform candidates, reruns the label gate, builds the immutable distillation
artifact, and starts full training. The second waits for
the completed checkpoint and runs the complete frozen validation/test physics set
as four deterministic hash-balanced CUDA shards with bounded process retries. The
final report includes lower-tail improvement, upper-tail teacher gap, worst learned
loss, and maximum Nt ratio in addition to the pilot gates.

The separately frozen post-gate circle suite has 120 conditions crossing five
positions, three radii below/inside/above the training range, two material regimes,
two unseen angles, and exact 32/48 budgets. The quadrant centers preserve at least
2.5 coarse cells between the largest circle and the PML. The restartable paired
learned/uniform runner and four-GPU continuation wait for the main frozen physics
gate before launching 240 solver cases. Passing this gate is now required before
progression to sparse scenes.

The first full position/scale execution is complete and records a real remediation
gate rather than promotion. Overall, the CNN wins by at least 5% on 92/120 pairs
(76.7%) with 1.667x median joint-score improvement, and every size/position median
gate passes. Two centered, lossless, radius-0.135 m uniform baselines at 32x32 remain
above the `1e-5` tail limit after 560 ns. The worst settled case is the translated
northeast, lossy epsilon_r=20, radius-0.135 m circle at 48x48, where CNN/uniform
improvement is 0.435x against the frozen 0.5 floor. The frozen gate therefore fails
on settling and worst-case accuracy. Do not weaken it: extend the declared settling
schedule and add disjoint translated/scaled training lineages before rerunning the
original OOD suite. The initial monitor feasibility defect was corrected separately
with a declared 0.12 m PML and the widest legal per-grid NF2FF contour; all 240 axes
pass preflight under that policy.

The first disjoint translated/scaled remediation added 192 training conditions and
the fresh combined fit passed the original 448-case validation/test physics gate.
The rerun position/scale suite also settled all 120 pairs and improved 99 by at least
5%, with a 1.622x overall median. One large lossy northeast case at 48x48 remains
below the frozen worst-case floor: 0.389x versus the required 0.5. The exact frozen
condition remains excluded. A second disjoint remediation envelope therefore samples
large lossy circles densely around all four corners and incidence angles bracketing
the failed regime before the sparse-object curriculum proceeds.

The second disjoint corner-focused remediation is complete. Its fresh combined fit
again passes the original 448-case frozen validation/test physics gate with all cases
settled. The unchanged 120-case circle position/scale suite now also passes every
frozen check: all pairs settle, 93/120 improve by at least 5%, median joint-score
improvement is 1.664x, and the worst improvement is 0.568x against the fixed 0.5x
floor. This promotes the project to the sparse curriculum without weakening a gate.

M4.3A implementation is underway as a restartable localized two-dielectric-cylinder
headroom pilot. Eight scenes span close/moderate/wide gaps, unequal radii and
materials, cluster translations/orientations, and incidence angles while retaining
0.5--12% occupied area and less than 45% projected support per axis. Fine uniform
192/256 references are convergence-checked before 168 exact 32/48/64 candidate
simulations. Candidate policies include uniform, multi-interface, cluster, gap, and
hybrid density placement. M4.3B is now explicitly reserved for circle--rectangle,
rectangle--rectangle, dielectric--PEC, and PEC--PEC pair strata before moving to
3--4 object clusters.

M4.3A is now complete and passes its predeclared headroom gate. All 16 uniform
192/256 references settle, with cross-resolution joint losses from 1.0e-5 to
4.3e-4. All 168 exact-budget candidate cases also settle. Across the 16 low-budget
32/48 scene-budget groups, 12 (75%) improve by at least 5% and median soft-Nt
improvement is 1.402x. Close, moderate, and wide-gap strata pass separately at
66.7%, 100%, and 66.7% meaningful-win fractions. This promotes M4.3B mixed-shape
and mixed-material pair implementation.

The planned model ablation is expanded to all nine combinations of 128/256/384
input resolution and 16/24/32 U-Net base width. A direct FP32 benchmark on a
24-GiB TITAN RTX measured 1.18--3.94 GiB peak allocation per job at the selected
microbatches. With four identical GPUs, all nine fits can run concurrently with
resolution-specific gradient accumulation and a common effective batch size. The
input schema remains fixed across the grid; network width is the ablated channel
count.

M4.3B now also passes its predeclared mixed-pair headroom gate. Its 16 uniform
192/256 references and all 168 exact-budget candidates settled. Reference
cross-resolution joint losses range from 1.4e-5 to 1.2e-3. At 32/48 cells,
11/16 scene-budget groups improve by at least 5%, with 1.356x median soft-Nt
improvement. Circle--rectangle and rectangle--rectangle shape strata and each
dielectric--dielectric, dielectric--PEC, and PEC--PEC material stratum pass their
separate meaningful-win thresholds. All 184 saved cases were verified against the
phase-specific source fingerprints reconstructed by
`scripts/recover_sparse_mixed_provenance.py` after CNN-only source edits changed the
package-wide hash. The continuous PEC objects in this pilot are rectangles.

The first shared sparse-input ablation dataset, `sparse_joint_4e1cc5513e1b1335`,
contains 1,936 historical single-circle examples and 48 new pair examples, split by
entire sparse scene into 30 train, 6 validation, and 12 test conditions. Missing
candidate policies are masked. Nine fresh U-Nets at all 128/256/384 by 16/24/32
combinations have been launched concurrently on four GPUs under
`runs/sparse_nine_model_grid/launch.json`. Every fit uses the same seven physical
raster channels, 15 conditioning values, saved complex-field/RCS-ranked mesh targets,
seed, optimizer, and effective batch size 96. The sampler gives sparse scenes 70%
and simple controls 30% of training draws. This small sparse set is an initial
capacity/resolution ablation; repeated draws do not create new geometry diversity.

The follow-up scene-diverse campaign manifest, `sparse_pairs_4ce1a2f1820b2c52`,
has 96 preflighted scenes: 24 each from two-dielectric circles, two-dielectric
mixed shapes, dielectric/PEC pairs, and two-PEC rectangles. Each family contributes
18 train, 3 validation, and 3 test scenes, giving 72/12/12 overall. It spans 38
close, 34 moderate, and 24 wide-gap scenes; the smallest gap is 12.49 mm, about
2.7 cells at the 256-cell fine reference grid. Every exact-budget candidate and
192/256 reference grid passed grading, NF2FF monitor, and conformal-PEC preflight.
Its 192-case reference phase has been launched alongside the nine training fits.
Run `PYTHONPATH=. .venv/bin/python scripts/sparse_grid_status.py` for a compact
snapshot of all nine epochs and accepted sparse reference counts.

## Next work

1. Broaden PEC enlargement validation to close gaps, cavities, resonances, strong
   grading, mixed dielectrics, and split-edge thin screens. Compare accuracy/cost
   before choosing a default. Plain conformal remains the current default.
2. Resolve observation/CPML convergence across the benchmark range; add the
   openEMS-style TFSF comparison and an external solver cross-check.
3. Generalize the implemented adaptive policy from the analytic-cylinder runner to
   the new scene schema. Extend mixed coupling to declared contact/overlap priority
   only after deriving its operator. NF2FF is currently a post-step host calculation.
   Float32 is accurate but has no throughput benefit here, so retain float64.
4. Let the live 9,312-case exact-budget campaign and its automated training and
   frozen-physics handoffs finish. Audit the resulting full report, then expand to
   sparse and complex families only after the simple-family gate passes.
