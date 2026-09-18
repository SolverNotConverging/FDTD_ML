# Current implementation progress

Status date: 2026-09-18.
Implementation baseline: Git commit `99374e7`
(`Implement stage-one CUDA TMz solver and CNN-driven exact-budget mesher`).

The two stage-one implementation tasks are complete and validated: the PEC-boundary
nonuniform CUDA TMz solver and the CNN-to-FDTD mesher. A trained meshing model,
CPML, and the subsequent dataset/training pipeline are not yet implemented.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full roadmap,
[README.md](README.md) for usage, and [docs/validation.md](docs/validation.md)
for the numerical and performance evidence.

## Stage status

| Stage | Status | Evidence or remaining work |
|---|---|---|
| 1A: nonuniform CUDA TMz solver | Complete for PEC scope | Compiled native runtime; real-GPU and analytical tests |
| 1B: CNN-to-FDTD mesher | Complete for inference scope | Exact-budget mesher; conditioned CNN; checkpoint-to-CUDA tests |
| 2: CPML and dataset simulation conventions | Not started | Fixed collars, auxiliary state, reflection tests, source/probe comparison conventions |
| 3: procedural datasets and converged references | Not started | Scene/data schemas, reference convergence, split design, physical metrics |
| 4: teacher imitation | Not started | Teacher targets, training loop, trained checkpoint |
| 5: physics targets and distillation | Not started | Candidate generation, FDTD scoring, Pareto selection, distillation |
| 6: generalization and optional surrogate | Not started | OOD studies, ablations, optional surrogate validation |

## Completed: environment and repository

- Created a Python 3.12 project environment in `.venv` with `uv`.
- Installed NumPy, SciPy, Cython, CUDA-enabled PyTorch, pytest, and Ruff.
- Added `pyproject.toml`, `.python-version`, and the reproducible `uv.lock`.
- Verified `uv sync --locked --python 3.12`.
- Added a Windows build script that discovers MSVC and compiles/links the Cython
  extension and CUDA runtime. Resolved compiler and Windows SDK path issues on this host.
- Initialized Git and committed the implementation. Environment files, generated
  native binaries, caches, artifacts, model checkpoints, and IDE settings are ignored.
- Used the previous local FDTD library as an API/numerical reference without modifying it.

## Completed: solver

- Continuous material/geometry scene, independent of the simulation grid.
- Isotropic nondispersive epsilon/mu, electric loss, vacuum, and exact PEC masks.
- Rectangles, circles, triangles, simple polygons, and anchored thin PEC lines.
- Nonuniform Yee updates with primal/dual spacings and conservative CFL validation.
- Float32 and float64 CUDA execution behind a validated Cython/C++ boundary.
- Native timestep loop with device-resident fields, coefficients, source waveforms,
  source/receiver indices, and receiver histories; no Python stepping loop.
- PEC outer boundaries, point/line soft sources, Gaussian and sinusoidal waveforms.
- Point/line Ez receivers, final fields, physical sampling times, and post-run DFT.
- `Nt` or fixed physical `t_end`, cell-update cost, CUDA-event timing, wall timing,
  and upload/download/residency diagnostics.
- Deterministic handling of overlapping sources, repeatable fresh runs, input
  validation, and resource cleanup.
- Explicit NumPy oracle for numerical tests; no silent CPU production fallback.

Main code: [simulation.py](src/fdtdmesh/simulation.py),
[scene.py](src/fdtdmesh/scene.py), [sources.py](src/fdtdmesh/sources.py),
and [solver/](src/fdtdmesh/solver/).

## Completed: mesher and CNN inference

- Positive density integration, deterministic capped largest-remainder cell allocation,
  and density-quantile line placement.
- Exact Nx/Ny budgets, exact physical boundaries, and mandatory anchor retention.
- Optional minimum/maximum spacing and adjacent-cell grading projection.
- Explicit rejection of infeasible, duplicate, or unresolvable mesh lines.
- Raw scene rasterization with epsilon, conductivity, PEC, sources, receivers,
  x/y anchors, and mu; consistent pixel tie-breaking under physical scaling.
- Fully convolutional residual U-Net with multi-scale FiLM conditioning.
- Electrical-size, frequency-ratio, and budget conditioning; positive axis densities
  through normalized log-sum-exp pooling and softplus.
- Self-describing versioned checkpoints, compatible-model loading, and metadata checks.
- Public `mesh_uniform`, `set_mesh`, `mesh_from_density`, `load_mesh_model`, and
  `mesh_with_model` entry points.
- End-to-end checkpoint -> CNN -> constrained mesh -> CUDA simulation example.

Main code: [mesh.py](src/fdtdmesh/mesh.py) and [ml.py](src/fdtdmesh/ml.py).
No trained weights are supplied; the default CNN example explicitly uses random weights.

## Validation already performed

The last implementation validation recorded **46 passing tests, no skips**, including
real-CUDA tests, in 2.69 seconds. Ruff lint and formatting checks passed. These are
the recorded implementation results, not a new test run for this documentation update.

Validated hardware/toolchain: Windows, RTX 4070 Laptop GPU (8 GB), NVIDIA driver
596.49, CUDA toolkit 13.3, MSVC 14.51.36231, Python 3.12.14,
PyTorch 2.14.0+cu130, NumPy 2.5.3, and SciPy 1.18.1.

Coverage includes:

- CUDA/NumPy agreement on uniform/nonuniform grids in both precisions, including
  dielectric/magnetic loss, PEC objects, duplicate sources, and line receivers.
- Analytical cavity convergence in vacuum and a lossy magnetic material; errors
  decrease by approximately four when grid spacing is halved for these smooth cases.
- Exact PEC shielding, zero-state preservation, repeated-run agreement, and malformed
  native-input rejection.
- Exact mesh budgets, anchors, density quantiles, constraint projection, randomized
  mesh invariants, checkpoint compatibility, CNN gradients, and scale consistency.
- Full CNN-to-CUDA integration and zero reported host/device transfers while stepping.
- Successful uniform, nonuniform, and CUDA-CNN examples.

A fixed-duration 128 x 96 benchmark measured median GPU times of 69.1 ms for
2,857 uniform-grid steps and 100.6 ms for 4,406 nonuniform-grid steps over the
same requested 1 ns duration. This demonstrates the CFL/work penalty of smaller
cells; it is not evidence that the untrained CNN improves mesh quality.

## Current limitations and deliberate scope choices

The [anchor grading audit](docs/anchor_grading.md) confirms that explicit ratio
constraints move neighboring lines, including across anchors. Grading is currently
off by default, and the fixed interval allocation can prevent recovery of an otherwise
feasible graded mesh. The audit adds a reproducible plotting example; it does not
change mesher behavior.

- PEC outer boundaries only. `add_PML()` raises `NotImplementedError`; no fixed
  PML collar or CPML auxiliary update exists yet.
- Geometry is point-sampled at Yee locations. There is no subpixel averaging, and
  curved interfaces have staircase error. Unresolved unanchored features can disappear.
- A uniform mesh rejects anchors it cannot represent; the density/model mesher is
  required in those cases. New anchors invalidate an existing mesh until remeshing.
- Grading projection fixes the integer cell allocation between anchors. It may reject
  an allocation even when another integer allocation would be feasible.
- Soft sources add Ez per step. Excitation normalization across changing dt needs
  an explicit convention before physics-based candidate ranking.
- Point receivers snap to nodes; comparisons across meshes need a consistent
  physical receiver convention. Actual sampled coordinates are returned.
- Line monitors record Ez only. Spectra are computed after the run on CPU; no
  online GPU DFT or energy monitor is implemented.
- Full waveforms and receiver histories require memory proportional to Nt.
  Batching, bounded-history recording, and CUDA graph replay are not implemented.
- Residency evidence is runtime accounting plus source inspection, not an independent
  Nsight/CUPTI transfer trace.
- Checkpoints must match the current architecture and metadata contract; older models
  may need conversion. The eight-channel raster includes raw mu, and pooling uses
  log-mean-exp to remove constant-field raster-size dependence.
- Windows is validated; the provided Linux build path has not been validated.
- No procedural training dataset, converged arbitrary-scene reference generator,
  teacher training, physics distillation, or demonstrated learned improvement exists yet.

## Next implementation milestone

Proceed to Stage 2: reserve uniform-normal PML collars within the exact total budget,
anchor their interfaces, implement device-resident CFS-CPML, and quantify reflection
and stability. In parallel in the roadmap, define fair excitation and receiver
comparison conventions before building reference datasets. Follow the detailed
acceptance criteria in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Reproduce the existing validation

From the project root in PowerShell:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
uv sync --locked --python 3.12
.\scripts\build_cuda.ps1
New-Item -ItemType Directory -Force artifacts | Out-Null
.venv\Scripts\python.exe -m pytest -q --basetemp=artifacts/pytest
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\python.exe examples\simple_uniform.py
.venv\Scripts\python.exe examples\simple_nonuniform.py
.venv\Scripts\python.exe examples\cnn_mesh.py
```

Keep this file updated as stages are implemented. Record fresh test results and
commit references, and move work to complete only when its acceptance evidence exists.
