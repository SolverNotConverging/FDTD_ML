# ScatterMesh: mesh CNN v2

Learn nonuniform tensor-product meshes for **2D TMz scenes with nonuniform electromagnetic resolution demand**, using optimized teachers. The goal is better accuracy at equal cell count, or fewer cells at comparable accuracy, from simple primitives through composite engineering-like silhouettes and distributed mixtures. Low material occupancy is one useful case; highly occupied heterogeneous materials can also benefit.

**C0 CNN training is active.** Acquisition completed and passed the qualified-label gate with 68 training and 10 validation lineages. The first large-model checkpoints are saved; validation FDTD selection is queued after training finishes. The C0 test split remains untouched.

The sole project root is `/home/s2307298/projects/FDTD_ML`. New outputs belong in `runs_v2/`; original projects and previous planning documents remain in `archive/`.

Runnable geometry demonstrations are in [examples/](examples/README.md). Each loads a compatible trained mesh CNN, runs the compiled CUDA solver, and saves mesh and polar 2D scattering-width figures. `--compare` adds a converged finer uniform reference and a uniform mesh at the CNN's cell budget.

## Current plan

The [project-wide implementation plan](IMPLEMENTATION_PLAN.md) restarts cumulative learning at C0 with a fresh large CNN and more training data. Every sampled training/validation geometry, material assignment, illumination, and budget receives its own optimized teacher.

- C0–C2: primitives, position/scale, rotation/aspect ratio, corners and concavity; retain these as manuscript experiments.
- C3–C5: mixed pairs, qualified close gaps, then 3–10 sparse objects with controlled arrangement.
- C6–C7: holes, cavities, thin sections, internal material interfaces, and multiscale composite bodies.
- C8–C9: bird, warship, aircraft, satellite, and radio-telescope silhouettes, followed by sparse mixtures with different materials.

All stages measure how resolution demand is distributed. Develop entropy/effective-support descriptors of a material/interface/feature demand map, together with absolute demand and x/y demand coverage. Material coverage and material-histogram entropy alone are insufficient: the same materials can form one block or many fine layers. Include low-occupancy scenes, highly occupied heterogeneous regions, and uniformly demanding controls. Keep difficult layouts and uniform-winning cases; our tensor-product grid refines whole rows and columns.

Lossless high-permittivity cases remain lossless when they settle. Added loss used solely for convergence is allowed only after the original condition fails its declared maximum duration, and creates a separately identified material variant. Prescribed lossy materials, such as a bird surrogate, are input physics. Lossy and lossless comparisons use their own matching references and baselines.

The C0 restart is underway. Calibration qualified 14 of 18 primitive references with the extended spatial rule, while unresolved and unsettled cases remain excluded. The corrected 128-lineage acquisition is complete; 301 qualified train/validation examples are feeding fresh CNN training. The test split is untouched. Balanced batches may grow toward a proposed 512 C0 lineages as measured cost and learning curves permit. The previous 64/16/24 compact experiment and its 2.92-hour projection do not describe this expanded curriculum. See [the C0 restart protocol](docs/v2_pilot.md).

## Implemented foundation and remaining work

The compiled CUDA backend, continuous primitive/polygon geometry, separated-object scenes, exact-budget density projection, CNN, distillation, and bounded teacher optimizer exist. An eight-scene GPU development profile, optimized-teacher calibration, and the first 128-lineage acquisition completed. Development teacher gains are documented in [PROGRESS.md](PROGRESS.md); the large C0 model is training and no frozen physical test result exists yet.

The current generator covers aircraft/ship/vehicle outlines. Bird, satellite, telescope families, general internal material partitions, demand-entropy descriptors, and later-stage scheduler work remain planned. A truly material-filled background also needs a consistent source/PML/far-field extension; the current incident wave and far-field transform assume vacuum. Use `python -m scattermesh.c0_campaign_v2` for C0 manifests, workers, datasets, and training. Existing `launch` and `launch-compact` commands implement earlier curricula.

The main residual U-Net has widths 64/128/256/512/512, eight-head bottleneck attention, and **26,850,497 parameters** with its current nine 512×512 inputs. Outputs are positive x/y densities projected to exact cell counts and adjacent-cell ratios at most 3. Internal material interfaces will require a versioned input encoding and memory check. The weak dt exponent remains **0.05**. The width-16 comparison has 1,683,761 parameters.

## Required FDTD backend

**Every new FDTD profiling, reference, teacher, validation, and test simulation uses the Cython-bound, nvcc-compiled cooperative CUDA kernel.** CPU geometry/coefficient preparation precedes launch; field updates, CPML, conformal PEC, source, DFT, and field checks stay on GPU for the whole time loop with zero host/device transfers during stepping. CPU final processing is allowed. Legacy CPU/Torch solvers are restricted to parity regressions and saved comparisons. Torch remains available for CNN training.

Rebuild after changes to CUDA/Cython sources:

```bash
.venv/bin/python setup_cuda.py build_ext --inplace
```

The runtime rejects missing or stale builds. Three paired TITAN RTX benchmarks measured end-to-end speedups of 66.6× for a 96² dielectric circle, 35.8× for a 32² conformal PEC rectangle, and 43.4× for a 64² distributed dielectric scene. Maximum paired complex far-field difference was `3.01e-15` relative L2. See [benchmark records](runs_v2/cuda_fdtd_benchmark_v1/report.json).

The host has four TITAN RTX GPUs, driver 595.45.04, Torch `2.14.0+cu130`, and a CUDA 13.2 build targeting `sm_75`. GPU work requires host-device access in this execution context. The active `.venv` and editable installation resolve this project locally.

## Project map

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): authoritative C0–C9 stages, materials, sparsity, teachers, learning, and evidence.
- [PROGRESS.md](PROGRESS.md): current capabilities, measured evidence, pause state, and remaining implementation.
- [docs/v2_pilot.md](docs/v2_pilot.md): first C0 calibration and acquisition protocol.
- `src/scattermesh/`: numerical solver and geometry, optimization, learning, and evaluation modules.
- `runs_v2/`: completed development artifacts and future versioned campaign outputs.
- `archive/scattermesh_2026-09-23/` and `archive/receiver_cnn_2026-09-22/`: original projects.
- `archive/plan_revisions/2026-09-23_before_sparse_curriculum_restart/`: preceding planning documents.

The root `runs/` path is a local compatibility link into the scattering archive. Historical manifests preserve provenance; importers resolve artifacts locally without the former Codex worktree.
