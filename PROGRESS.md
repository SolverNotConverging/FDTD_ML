# Current project progress

Updated 2026-09-19. **Stages 1–4 are implemented**, including mandatory 1.4 grading,
CUDA CPML, procedural datasets and convergence-gated evaluation. Stage 3 reference
coverage remains partial. Generator v4 retains the low-dk-dominant material mixture
and randomized source/receiver coverage while adding 32x32 and 48x48 budget strata.
The evaluator supports longer resonance windows and now has a reference-only,
restart-safe 64/16/32 train/validation/IID workflow. Stage 4 supplies projected
teacher targets, reproducible/resumable training, and held-out CUDA evaluation.
Stage 5 now includes a second candidate search, two early-stopped continuation
branches, measured validation selection, a selected 80%-physics checkpoint, and a
measured four-budget continuation experiment.
This record accompanies the version-0.5.0 imitation-training update. Earlier commits:
`99374e7` (stage one), `2437779` (plan/progress), `085cd1b` (historical grading audit),
`e35a079` (stage two), `079b62c` (initial stage three), `00db055` (generator v2),
`1d3eca1` (generator v3), and `70c1a02` (Stage-4 implementation).

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
| 4: teacher imitation | Complete | Projected targets; resumable GPU training; trained checkpoint; IID/CUDA evidence |
| 5: physics targets and distillation | Four-budget pilot complete | 32/48/64/96 targets; FDTD-selected default retained; 63-pair IID evaluation |
| 6: generalization / optional surrogate | Not started | OOD studies, ablations and any surrogate validation |

## Previous update: Stage-4 teacher imitation

- Added versioned heuristic-teacher targets. Raw heuristic density is projected through
  the exact legal mesher before rebinning, so targets include budgets, anchors, PML
  collars, and mandatory grading.
- Added seeded AdamW training, validation, detached repair loss, bounded projection
  diagnostics, safe resume, best-checkpoint selection, dataset/hash/Git provenance,
  held-out evaluation, and real CUDA teacher-agreement checks.
- Trained width-16 ResU-Net for 40 epochs on 64 train samples and selected epoch 40.
  Best validation CDF loss is 2.621e-5. Held-out IID loss is 3.783e-5 versus 5.634e-4
  untrained (14.9x lower); all 64 tested projections are legal.
- Across 16 held-out CUDA scenes, the trained model improves waveform and spectrum
  teacher agreement in 11. Median errors improve (waveform 5.50% to 3.45%, spectrum
  6.05% to 4.78%), while one sensitive case makes means worse. Small density/line
  differences therefore do not guarantee small EM error.
- Updated root `AGENTS.md`: bounded repetitive work with objectively checkable output
  may use `gpt-5.6-luna`; numerical reasoning, architecture and integration stay with
  the primary/stronger model.

Validation: **113 tests passed, no skips**, including CUDA (16.41 seconds). Ruff
lint/format and locked offline uv synchronization passed. See
[Stage-4 evidence](docs/stage4_training.md) and the local ignored artifacts under
`artifacts/stage4/`.

## Current update: converged-reference corpus infrastructure

- Added exact-count generation for selected splits and the canonical 64 train,
  16 validation, and 32 untouched IID test manifest command.
- Generated the local 112-scene seed-2026 manifest with dataset ID
  `f17978780a329ec23e184b889440c52e5904e737d122d3a6d86bc5ae58abf7d7`.
  It has exact split counts, 1–8 primitives, epsilon_r 1.020–29.128, sigma
  0–9.429 S/m, and source coverage of approximately 0.19–0.81 on both axes.
- Added a reference-only runner that commits each scene independently, resumes only
  under an identical dataset/configuration contract, and updates JSON/Markdown
  coverage after every scene. Completed failures and nonconvergence remain recorded.
- Each scene retains its own adaptive duration, duration history, accepted spatial
  budget, latest waveform/spectrum/mesh, cost diagnostics, and wall time. Only
  `converged` is accepted as ground truth.
- Kept direct Yee-location material point sampling. No averaging or subpixel model
  was introduced.

The 112-scene manifest is generated and validated. The full CUDA convergence sweep
has not yet been run, so accepted-reference counts are not claimed here. See
[the reference workflow](docs/reference_dataset.md).

Validation: **128 tests passed, no skips**, including CUDA (17.19 seconds). Ruff
lint/format and locked offline uv synchronization passed. A separate three-scene
reduced-grid CUDA smoke run verified converged, nonconverged, failed, and resume
paths; it is orchestration evidence rather than production reference data.

## Current update: first substantial Stage-5 cycle

- Completed all 112 reference attempts: 30/64 train, 6/16 validation, and 16/32
  untouched IID scenes converge. There are 52 accepted references, 56 explicit
  spatial nonconvergences, and four resource failures.
- Evaluated eight candidate families on both budgets of every accepted train and
  validation scene. Generated 72 finite, hash-verified physics targets: 60 train and
  12 validation. Uniform wins 23 targets; 49 select a nonuniform family.
- Trained a 20-epoch physics-distilled checkpoint from the Stage-4 model. Epoch 16
  reduces validation target loss from 1.586e-4 initially to 1.161e-4.
- Evaluated uniform, heuristic, Stage-4 imitation, and Stage-5 distilled meshes on
  16 converged untouched IID scenes and two budgets, with 128/128 successful runs.
  Stage 5 improves median max EM error from 4.690% to 4.328% versus Stage 4 and wins
  17/32 pairs. Its mean is slightly worse, so sensitive outliers remain.
- Against the heuristic, Stage 5 has lower paired error in 17/32 and lower work in
  28/32; median paired ratios are 0.969 error and 0.864 work.

See [Stage-5 evidence](docs/stage5_training.md). Generated data and checkpoints are
under ignored `artifacts/reference_pilot/` and `artifacts/stage5/`.

Validation: **131 tests passed, no skips**, including CUDA (16.80 seconds). Ruff
lint/format, locked offline uv synchronization, checkpoint/target hashes, and the
rendered Stage-5 summary figure were verified.

## Current update: Stage-5 available-data continuation

- Stopped the expanded local reference run after 172/704 train decisions so the
  expensive corpus can move to servers. It preserves 79 converged and 93 rejected
  scenes, with no failures. Five promising rejects were separately recovered at
  2048; one of four former resource failures converged at 1024, while the other
  three completed as retained nonconverged cases under the larger limit.
- Repeated candidate search on all 36 converged pilot train/validation references.
  The 72 selected targets include current Stage 5, Stage 4, uniform, heuristic,
  mixtures, and perturbations. All rejected reference scenes remain reported.
- Continued from the Stage-5 checkpoint with 50% and 80% physics weights and a
  60-epoch cap. Early stopping selected epoch 11/21 for 50% and epoch 7/17 for 80%.
- Actual validation FDTD results selected the 80% checkpoint despite its worse CDF
  loss: median EM error is 4.934% versus 5.880% for 50%, with lower median work.
- On 32 untouched IID scene-budget pairs, the selected model achieves 4.068% median
  and 4.902% mean maximum EM error at 26,686,976 median cell updates. This improves
  the prior Stage-5 aggregate median, mean, and median work. Every one of the 128
  strategy runs succeeded.
- Generator v4 adds 32x32 and 48x48 to the existing 64x64 and 96x96 ordinary
  budgets. Budget-OOD remains 80x80 and 112x112. Low-budget infeasibility stays an
  explicit outcome under unchanged anchors, collars, and 1.4 grading. Current-CNN
  mesh and cell-width plots were rendered across all four budgets.

See the continuation section in [Stage-5 evidence](docs/stage5_training.md). Local
artifacts are under ignored `artifacts/stage5_available/`; the rendered summary was
visually inspected.

## Current update: four-budget Stage-5 continuation

- Added explicit square-budget overrides to teacher generation, physics search, and
  evaluation while preserving the original manifest and reference-corpus hashes.
- Generated 308 feasible heuristic pairs and retained 12 infeasible pairs. The 36
  converged train/validation references yielded 142 physics targets: 34 at 32x32
  and 36 each at 48x48, 64x64, and 96x96.
- Continued the selected model with an 80% physics blend. Early stopping selected
  epoch 24 and stopped at epoch 34. Validation median/mean errors improved from
  9.723%/23.764% to 7.300%/21.845%.
- On 63 feasible held-out pairs, the continuation wins 35 paired comparisons and
  improves median error at 32x32, 48x48, and 64x64. Aggregate median/mean errors are
  7.507%/12.855% versus 7.044%/12.034% for the prior model, so the current default
  checkpoint remains selected.
- Rendered and visually checked the per-budget FDTD comparison, graded meshes, and
  cell-width profiles under ignored `artifacts/stage5_low_budget/`.

Validation: **138 tests passed, no skips**. Ruff also passes.

## Previous update: dk mixture and spatial probe coverage

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

- A trained teacher-imitation checkpoint now exists locally, but it does not demonstrate
  electromagnetic improvement over the heuristic or uniform meshing. Reference
  convergence is empirical for selected observables;
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

Move large reference generation to server hardware and expand train/validation
coverage across 32x32, 48x48, 64x64, and 96x96, especially for 32x32 outliers and
compositional scenes. Examine staircase convergence, PML sensitivity,
and sampling bandwidth. Compare strategies within equal budget strata and do not
weaken the reference gate merely to increase dataset size; retain failures and
infeasible scene-budget pairs in coverage reports.
