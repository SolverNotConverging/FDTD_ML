# ScatterMesh: mesh CNN v2

This project is a proof of concept for CNN-generated nonuniform tensor-product meshes in 2D TMz conformal FDTD. The main evidence is **better scattering accuracy at a fixed spatial cell count, or fewer cells at comparable accuracy, on unseen engineering-like silhouettes and distributed sparse scenes**.

The sole working root is `/home/s2307298/projects/FDTD_ML`. New datasets, checkpoints, logs, and reports belong under `runs_v2/`; previous projects remain in `archive/`.

## Project direction

The [project-wide implementation plan](IMPLEMENTATION_PLAN.md) covers C0–C9 with implementation instructions, numerical checks, and deliverables for every stage. The first demonstration uses C0–C2 foundation data and ordinary C3/C5 separated objects, then evaluates bounded C8/C9 target scenes. Extreme gaps, holes/cavities, very thin details, and larger multiscale scenes are subsequent extensions.

The initial compact experiment targets approximately 64 new training lineages, 16 validation lineages, and 24 frozen target scenes: 12 individual silhouettes and 12 distributed scenes. It reuses qualified historical training data, reduces teacher-search work, and compares uniform, a fixed geometry-based policy, and the CNN. Counts and reference rules are finalized using six to eight development cases. Full-scale 3D engineering simulations are outside this proof of concept.

## Current status

**The revised project plan is documented; the executable campaign still needs adaptation.** Existing v2 scene inputs assume one object, and the runner still enforces the superseded 128/256-lineage C0–C2 campaign and its fixed resource/reference rules. Its saved `sizing_gate_stopped` result applies to that earlier design. It is not a runtime estimate for the revised demonstration.

Available foundations include the TMz solver, continuous single-object geometry, mesh projection, scoring, CNN, training/evaluation framework, and a portable import of 1,104 historical training examples. No v2 CNN has been trained, and no C8/C9 test result exists. See [current progress](PROGRESS.md) for implemented capabilities and remaining work, and [the first-demonstration protocol](docs/v2_pilot.md) for experiment details.

The main model is a five-level residual U-Net with widths 64/128/256/512/512, eight-head bottleneck attention, and **26,850,497 parameters** in the current implementation. Its nine 512×512 image inputs are independent of the much smaller FDTD grid budgets. Outputs are positive x/y densities projected to exact requested cell counts and adjacent-cell ratios at most 3. The weak time-step exponent remains **0.05**. The 1,683,761-parameter comparison model is a later ablation after the main demonstration.

## Local checks

The active `.venv` and editable package installation point to this directory. From the project root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_local_archive.py
.venv/bin/python scripts/run_mesh_cnn_v2.py --help
```

The existing `launch` action runs the superseded C0–C2 workflow. It must be updated for multiple objects, compact manifests, adaptive reference qualification, and the fixed heuristic baseline before it can execute the revised plan. The archive verifier checks all relocated links and representative file hashes; `--full` hashes every inventoried archive file.

## Project map

- `IMPLEMENTATION_PLAN.md`: authoritative C0–C9 roadmap, first demonstration, and subsequent stage instructions.
- `PROGRESS.md`: measured evidence, implementation status, and next work.
- `docs/v2_pilot.md`: compact demonstration protocol and required changes to the executable workflow.
- `src/scattermesh/`: numerical solver and current geometry, candidate, reference, dataset, CNN, training, and evaluation modules.
- `scripts/run_mesh_cnn_v2.py`: existing campaign entry point awaiting the revised protocol.
- `runs_v2/`: local v2 evidence and new outputs; earlier v2 profile records are retained as provenance.
- `archive/scattermesh_2026-09-23/` and `archive/receiver_cnn_2026-09-22/`: preserved previous projects and restoration records.

The root `runs/` path is a relative compatibility link into the scattering archive. Historical manifests remain unchanged; the importer resolves their artifacts locally. The former Codex worktree is absent.
