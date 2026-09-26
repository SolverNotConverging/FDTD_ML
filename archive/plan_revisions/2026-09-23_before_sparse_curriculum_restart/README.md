# ScatterMesh: mesh CNN v2

This project is a proof of concept for CNN-generated nonuniform tensor-product meshes in 2D TMz conformal FDTD. The main evidence is **better scattering accuracy at a fixed spatial cell count, or fewer cells at comparable accuracy, on unseen engineering-like silhouettes and distributed sparse scenes**.

The sole working root is `/home/s2307298/projects/FDTD_ML`. New datasets, checkpoints, logs, and reports belong under `runs_v2/`; previous projects remain in `archive/`.

## Project direction

The [project-wide implementation plan](IMPLEMENTATION_PLAN.md) covers C0–C9 with implementation instructions, numerical checks, and deliverables for every stage. The first demonstration uses C0–C2 foundation data and ordinary C3/C5 separated objects, then evaluates bounded C8/C9 target scenes. Extreme gaps, holes/cavities, very thin details, and larger multiscale scenes are subsequent extensions. Bounded CPU C3/C4 studies qualify selected dielectric circle-pair conditions. A matched-area C5 study found that current fixed geometry policies rarely help on three/five-object sparse layouts; learned transfer remains to be tested. Broader C3/C5 qualification and the C8/C9 demonstration remain open.

The initial compact experiment targets approximately 64 new training lineages, 16 validation lineages, and 24 frozen target scenes: 12 individual silhouettes and 12 distributed scenes. It reuses qualified historical training data and now optimizes teachers separately for each training or validation geometry, material, incidence angle, and cell budget before CNN distillation. The final comparison remains uniform, a fixed geometry-based policy, and the CNN. Full-scale 3D engineering simulations are outside this proof of concept.

## Current status

**No v2 CNN or frozen target-scene campaign has run yet.** The compiled CUDA backend and eight-scene GPU development profile are complete. The current cost projection, including 96 physics-scored optimizer trials for each of 144 train/validation conditions, is about 2.92 hours on four GPUs; it is a planning estimate, not a completed campaign. Fixed teacher policies were weak outside one circle. Bounded optimization improved a high-contrast pair, a PEC circle, and two development silhouettes, with the latter two gains persisting against finer independent references. A one-condition worker smoke saved nine valid candidates and built a CNN dataset. See [current progress](PROGRESS.md) and [the refreshed profile](runs_v2/compact_poc/profile/compact_profile.json).

Available foundations include exact continuous intersections, exact-budget mesh projection, the 26.85-million-parameter CNN, a portable import of 1,104 historical training examples, and GPU solver parity checks. The compact scene generator creates 64 training, 16 validation, and 24 test lineages. These targets are simplified procedural 2D outlines, not CAD designs. Development silhouette results do not count as frozen C8/C9 test evidence.

The main model is a five-level residual U-Net with widths 64/128/256/512/512, eight-head bottleneck attention, and **26,850,497 parameters** in the current implementation. Its nine 512×512 image inputs are independent of the much smaller FDTD grid budgets. Outputs are positive x/y densities projected to exact requested cell counts and adjacent-cell ratios at most 3. The weak time-step exponent remains **0.05**. The 1,683,761-parameter comparison model is a later ablation after the main demonstration.

## Required FDTD backend

**Every new profiling, reference, teacher, validation, and test FDTD run must use the compiled CUDA kernel.** The Cython binding in `src/scattermesh/_cuda_fdtd.pyx` calls a cooperative kernel compiled from `cuda_fdtd_kernels.cu` by `nvcc`. Geometry and coefficients are prepared on CPU before launch; fields, CPML state, PEC treatment, source updates, field checks, and DFT remain on GPU for the entire time loop. There are zero host/device transfers during time stepping. CPU NumPy and legacy Torch solver runs are reserved for numerical parity tests and the saved speed comparison; they must not produce new campaign labels or evaluation results. The run scripts reject CPU devices.

Build the kernel after any change to its Cython or CUDA source:

```bash
.venv/bin/python setup_cuda.py build_ext --inplace
```

The runtime checks source hashes against the local build record and stops if the extension is stale. The benchmark on TITAN RTX GPU 2 found median end-to-end speedups of 66.6× for a 96×96 dielectric circle, 35.8× for a 32×32 conformal PEC rectangle, and 43.4× for a 64×64 distributed dielectric scene, each over three paired repeats. Maximum paired complex far-field disagreement was `3.01e-15` relative L2. See [the raw benchmark](runs_v2/cuda_fdtd_benchmark_v1/report.json).

## Local checks

The active `.venv` and editable package installation point to this directory. From the project root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_local_archive.py
.venv/bin/python scripts/run_mesh_cnn_v2.py --help
.venv/bin/python setup_cuda.py build_ext --inplace
.venv/bin/python scripts/run_mesh_cnn_v2.py profile-compact
.venv/bin/python scripts/optimize_development_teachers.py --cells 48 --evaluations 96
.venv/bin/python scripts/run_mesh_cnn_v2.py launch-compact
```

The existing `launch` action still runs the superseded C0–C2 workflow. Use `profile-compact`, `freeze-compact`, and `launch-compact` for the revised large-model C8/C9 path. The measured computational cost gate passes, but no manifest has been frozen or full campaign launched. The archive verifier checks all relocated links and representative file hashes; `--full` hashes every inventoried archive file. On the host, driver 595.45.04 exposes four TITAN RTX cards. Torch `2.14.0+cu130` is used for CNN training; the FDTD kernel is compiled with CUDA toolkit 13.2 for `sm_75`. GPU commands need host device access in this execution context.

## Project map

- `IMPLEMENTATION_PLAN.md`: authoritative C0–C9 roadmap, first demonstration, and subsequent stage instructions.
- `PROGRESS.md`: measured evidence, implementation status, and next work.
- `docs/v2_pilot.md`: compact demonstration protocol and required changes to the executable workflow.
- `src/scattermesh/`: numerical solver and current geometry, candidate, reference, dataset, CNN, training, and evaluation modules.
- `scripts/run_mesh_cnn_v2.py`: legacy launch path plus compact profile, manifest, worker, large-model training, validation selection, frozen evaluation, and reporting orchestration.
- `runs_v2/`: local v2 evidence and new outputs; earlier v2 profile records are retained as provenance.
- `archive/scattermesh_2026-09-23/` and `archive/receiver_cnn_2026-09-22/`: preserved previous projects and restoration records.

The root `runs/` path is a relative compatibility link into the scattering archive. Historical manifests remain unchanged; the importer resolves their artifacts locally. The former Codex worktree is absent.
