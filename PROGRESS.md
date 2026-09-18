# Current project progress

Updated 2026-09-18. **Stages 1 and 2 are implemented and validated**, including
the agreed mandatory 1.4 grading policy and joint constrained mesh optimization.
This record accompanies the stage-two implementation commit. Earlier commits:
`99374e7` (stage one), `2437779` (plan/progress), `085cd1b` (historical grading audit).

See [implementation plan](IMPLEMENTATION_PLAN.md), [API and conventions](README.md),
[stage 2 validation](docs/stage2_validation.md), and [grading plots](docs/anchor_grading.md).

## Stage status

| Stage | Status | Evidence / remaining work |
|---|---|---|
| 1A: nonuniform CUDA TMz solver | Complete | Native float32/64 solver; oracle and analytical cavity tests |
| 1B: CNN-to-FDTD mesher | Complete for inference | Exact budgets/anchors; conditioned CNN; checkpoint-to-CUDA example |
| Grading policy revision | Complete | Mandatory <=1.4; joint anchor assignment and line L1 optimization; exhaustive small-case optimum test |
| 2: CPML and simulation conventions | Complete for vacuum collars | CUDA CPML; reflection controls; current normalization; physical receivers |
| 3: procedural datasets and converged references | Not started | Schemas, generation, convergence, splits and physical metrics |
| 4: teacher imitation | Not started | Teacher targets, training loop and trained checkpoint |
| 5: physics targets and distillation | Not started | Candidate generation, FDTD scoring, Pareto selection and distillation |
| 6: generalization / optional surrogate | Not started | OOD studies, ablations and any surrogate validation |

## Completed in this update

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

## Validation performed

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

- No trained meshing weights, training loop, procedural dataset, convergence-based
  reference generator, or demonstrated learned electromagnetic improvement yet.
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

Stage 3: versioned procedural scenes and reproducible train/evaluation splits,
convergence-checked references, and uniform/heuristic/CNN comparisons at common
physical excitation, observations and duration. Keep the CPML acceptance cases and
analytical cavity tests as independent gates while expanding the physics coverage.
