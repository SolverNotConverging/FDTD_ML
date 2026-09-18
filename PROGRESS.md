# Current project progress

Updated 2026-09-18. **Stages 1–3 are implemented**, including mandatory 1.4 grading,
CUDA CPML, procedural datasets and convergence-gated evaluation. Stage 3 reference
coverage remains partial. Generator v3 adds a low-dk-dominant material mixture to
the broad scene generator, with verified randomized source/receiver coverage.
The evaluator supports longer resonance windows.
This record accompanies the version-0.4.1 dk/probe update. Earlier commits:
`99374e7` (stage one), `2437779` (plan/progress), `085cd1b` (historical grading audit),
`e35a079` (stage two), `079b62c` (initial stage three), and `00db055` (generator v2).

See [implementation plan](IMPLEMENTATION_PLAN.md), [API and conventions](README.md),
[stage 2 validation](docs/stage2_validation.md), and [grading plots](docs/anchor_grading.md).

## Stage status

| Stage | Status | Evidence / remaining work |
|---|---|---|
| 1A: nonuniform CUDA TMz solver | Complete | Native float32/64 solver; oracle and analytical cavity tests |
| 1B: CNN-to-FDTD mesher | Complete for inference | Exact budgets/anchors; conditioned CNN; checkpoint-to-CUDA example |
| Grading policy revision | Complete | Mandatory <=1.4; joint anchor assignment and line L1 optimization; exhaustive small-case optimum test |
| 2: CPML and simulation conventions | Complete for vacuum collars | CUDA CPML; reflection controls; current normalization; physical receivers |
| 3: procedural datasets and converged references | Implemented; partial reference coverage | Versioned scenes, split validation, reference gates, three baselines, measured reports |
| 4: teacher imitation | Not started | Teacher targets, training loop and trained checkpoint |
| 5: physics targets and distillation | Not started | Candidate generation, FDTD scoring, Pareto selection and distillation |
| 6: generalization / optional surrogate | Not started | OOD studies, ablations and any surrogate validation |

## Current update: dk mixture and spatial probe coverage

- Generator v3 samples dielectric constants with 85% probability in 1–10 and
  15% in 10–30, logarithmically within each component. Configurable mixture settings
  are persisted in manifests and exposed through the generation CLI. High material-OOD
  values remain deliberately separate. This probability is a design default.
- Verified the existing randomized source and all three receiver positions. Added
  per-probe spatial coverage, quadrant and separation regression checks.
- A fresh 256-scene sample (32 per split, seed 2026) contains 267 dielectric objects
  across 96 train/validation/IID scenes; 88.76% have dk <=10, with dk spanning
  1.020–29.128. All four probe roles cover both coordinates approximately 0.19–0.81.
- Saved and visually checked `artifacts/stage3_v3/probes_and_permittivity.png`.
  Generator versions and historical manifests distinguish the changed seed realizations.
- Clarified that spatial convergence requires BOTH receiver waveform and complex
  DFT errors <=2% for two consecutive refinements. Time-window acceptance separately
  checks late waveform RMS/peak <=1% after excitation ends, extending if necessary.

Validation: **109 tests passed, no skips**, including CUDA (16.35 seconds). Ruff
lint/format and locked offline uv sync passed. Pytest used a workspace-local temporary
directory because the default shared temporary directory was inaccessible. No new
full reference sweep or training was performed. See [v3 evidence](docs/dataset_v3.md).

## Previous update: broader scenes and resonance windows (v2)

- Generator v2 randomizes counts, shape types, orientations and placement; ordinary
  scenes contain 1–8 objects, dense held-out scenes 9–12. Aspect ratios span 0.3–3.3.
- Logarithmic shape sizes span 3.5–50% of an axis before packing/resolution checks.
  Each dielectric object has independent epsilon_r (1.05–30) and conductivity
  (zero or 1e-5–10 S/m). Held-out material ranges extend further.
- Raster resolution increases to 128x128. Object/probe placement rejects unresolved
  gaps and unintended overlaps, and maintains probe clearance from geometry. Source/receiver
  positions and source pulse width/carrier are also randomized.
- Initial durations include a dielectric transit-time allowance. Reference defaults
  reach 1024x1024, with a 128-billion-update cap that can be raised explicitly.
- A separate 1% late-receiver RMS/peak gate doubles duration at most twice, restarting
  mesh refinement each time. Accepted references and every candidate use the same
  duration, temporal samples, frequencies and window. Sampling grows with duration;
  chunked spectral evaluation bounds temporary memory.
- New rejection status `time_unsettled` distinguishes persistent ring-down from
  spatial nonconvergence. Full duration attempts and an evaluated scene are saved.
- Frozen generator v1 remains available for reproducing earlier experiments.

Validation: **102 tests passed, no skips**, including real CUDA (15.59 seconds in the
recorded run). A reproducible 128-scene v2 sample covers all ordinary counts 1–8,
aspect ratios 0.331–3.207, geometry spans 0.0352–0.4661 of an axis, epsilon_r
1.10–27.87 and sigma 0–9.43 S/m. Gallery and distribution plots were inspected.
The harder four-scene CUDA sample still fails spatial/reference acceptance; one
scene extends from 2.93 to 5.85 ns and passes the ring-down check. This is broader
coverage, not a claim of learned generalization or universally converged labels.
Ruff lint/format checks and locked uv synchronization pass.
See [full v2 evidence](docs/dataset_v2.md).

## Completed in stage 3

- Added version-1 JSON scene/manifest schemas, content hashes, code/native-binary
  provenance, explicit physical excitation/observation contracts and resource limits.
- Added seeded geometry families and eight splits: training, validation, IID,
  composition, geometry, material, scale and unseen budgets. Enforced feature/gap
  resolution and cross-split lineage/near-geometry checks.
- Added uniform reference refinement with constant physical PML thickness and
  increasing collar resolution. Two consecutive waveform/spectral checks must pass;
  nonconvergence, solver errors and resource failures remain explicit.
- Added common-time interpolation, common Hann-window spectra, per-probe relative
  metrics, quiet-signal floors and gated circular phase errors.
- Added uniform-preference, material/edge heuristic and checkpoint-CNN evaluation,
  preserving exact budgets and fixed candidate collars. Random CNN use is labelled.
- Added CLI generation/evaluation, per-scene NPZ data and JSON status/metrics,
  Markdown accuracy/work/runtime reports, scene galleries and comparison plots.
- Updated the project to 0.3.0. No training loop or trained weights were added.

## Historical generator-v1 stage 3 validation

**93 tests passed, no skips**, including real CUDA tests (13.74 seconds in the
recorded full run). The 15 new tests cover reproducibility, schema validation,
manifest tampering, split leakage, raster feature/gap policy, reference PML refinement,
known amplitude/phase metrics, temporal alignment, consecutive convergence, failure
exclusion, resource caps and an end-to-end three-baseline GPU evaluation.

The seeded 32-scene experiment runs all eight splits. **15 references converge,
17 remain nonconverged, and all 90 candidate runs succeed**. The compositional split
has no accepted references in this initial experiment. This is a validation dataset,
not sufficient training or generalization evidence. Plots were visually inspected.
Full measured coverage and reproduction commands are in
[stage 3 validation](docs/stage3_validation.md).

## Completed in stage 2

- Made grading mandatory for density, model and supplied meshes; stricter limits
  are supported, while disabling grading or exceeding 1.4 is rejected.
- Replaced fixed interval allocation with joint mixed-integer anchor assignment
  and line-coordinate optimization. Minimized normalized mean absolute displacement
  from CNN density quantiles. Exact LP polishing retains anchors and ratio bounds.
- Distinguished optimization timeout/numerical failure from proven infeasibility.
  Added per-axis repair distance, solver status/gap and timing diagnostics.
- Added scale-invariant detached repaired-CDF loss for future CNN training, with
  gradients through predicted densities and explicit collar exclusion.
- Reserved immutable physical PML collars inside total budgets, ignored CNN collar
  density, anchored interfaces, and enforced interface grading. Continuous scene
  checks keep geometry, sources, receivers and interpolation support out of PML.
- Implemented CFS-CPML at electric and magnetic Yee positions, including corners,
  with four device-resident auxiliary arrays and fresh state per run.
- Extended the NumPy oracle and Cython/native interface; rebuilt the CUDA extension.
- Added integrated point-current and per-node current-density normalization with
  half-step sampling. Preserved explicit legacy field-increment behavior.
- Added exact-coordinate bilinear receivers, fixed-count line probes, common-time
  resampling without extrapolation, and documented spectral comparison conventions.
- Added the ninth raw PML input channel and version-2 checkpoint meshing policy;
  incompatible old checkpoints fail explicitly. Updated CNN-to-CPML example.
- Replaced the old grading plots with current-policy plots, retained the historical
  audit separately, and added CPML waveform/error/mesh plots and measured JSON.

## Historical stage 2 validation

Full suite: **78 passed, no skips**, including real GPU tests (12.67 seconds in the
recorded run). Ruff lint and formatting checks pass; `uv sync --locked --offline`
verified the version-0.2.0 environment. Uniform, nonuniform and CNN-to-CPML
examples all run successfully. Native CUDA build succeeded. Both precision modes agree with the
NumPy CPML oracle for x, y and xy absorption on nonuniform material scenes.

Independent boundary benchmark: eight combinations of 0/30/45/90-degree packets
and uniform/nonuniform meshes compared with an enlarged domain sharing the exact
interior mesh and dt. Peak waveform error ranges from 5.25e-6 to 7.07e-6, below the
predeclared 1% threshold. PEC controls give 25.19% and 27.12% peak error. At 1.5 ns,
fields remain finite and interior Ez norms are below 3e-5 of their initial values.
These measurements qualify the tested packets and parameters only.

Other coverage: exhaustive small-problem global mesh optimum, recovered allocation
counterexample, randomized legal meshes, fixed collar immutability, current
conservation across grids/dt, bilinear and time interpolation, checkpoint contract,
repair-loss gradients and scaling, repeated state reset, and zero stepping transfers.
The full suite retains the stage-one cavity convergence, PEC shielding, geometry,
source overlap and checkpoint-to-CUDA tests. Generated scientific figures were
visually inspected. See the validation document for commands and full measurements.

Environment: Windows, RTX 4070 Laptop GPU (8 GB), CUDA toolkit 13.3, MSVC 14.51,
Python 3.12 in `.venv`, CUDA PyTorch and SciPy from `uv.lock`. The previous library
at `C:\Users\Traveler\PycharmProjects\FDTD` was used as a reference and not modified.
Environment files, native binaries, plots, caches and random demo weights remain
ignored; source, tests, scripts, documentation and dependency metadata are tracked.

## Current limitations

- No trained meshing weights, training loop or demonstrated learned electromagnetic
  improvement yet. Reference convergence is empirical for selected observables;
  difficult scenes require more refinement/qualification. Dataset generation and
  comparison are implemented, with partial accepted reference coverage.
- Joint integer meshing can be expensive for large budgets/many anchors. A 30-second
  per-axis MILP search limit raises a distinct optimization error when unfinished;
  final LP refinement follows. Numerical tolerances and the normalized width floor
  are documented. Multiple optimal solutions may differ in symmetry.
- Grading bounds local ratios; it does not impose monotone spacing or curvature.
  Repair loss is an auxiliary consistency proxy and is not an EM objective.
- CPML collars must be vacuum. Broadband grazing incidence, extreme meshes and
  material interfaces near collars need further targeted qualification for datasets.
- Geometry remains point-sampled, with staircase errors and no subpixel averaging.
- Line monitors record Ez only. Full histories consume memory proportional to Nt;
  bilinear receivers record four node traces each. Batching, bounded histories,
  online GPU DFT/energy, graph replay and independent Nsight transfer profiling
  remain future work.
- Legacy field-increment sources are not physically invariant across dt; choose
  integrated current for point-source mesh comparisons. Align temporal coverage
  and spectral windows explicitly.
- Format-v1/eight-channel checkpoints require deliberate migration; loading them
  silently would violate the new raster and grading contract.
- Windows is validated. The provided Linux build path has not been validated.

## Next milestone

Stage 4: teacher-density targets and reproducible training over accepted scenes,
with held-out physical evaluation. Before larger experiments, improve reference
coverage (especially compositional scenes), expand seeds, and examine staircase
convergence, PML sensitivity and sampling bandwidth. Do not weaken the reference
gate merely to increase dataset size; retain failures in coverage reports.
