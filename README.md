# ScatterMesh: mesh CNN v2

This project predicts nonuniform tensor-product meshes for 2D TMz conformal FDTD scattering. It studies complex-field and scattering-width accuracy at identical spatial cell budgets and the cell savings needed to meet fixed error targets. The sole working root is `/home/s2307298/projects/FDTD_ML`; all new datasets, checkpoints, logs, and reports belong under `runs_v2/`.

## Status

The C0–C2 pilot framework is implemented, but its 24-hour bulk campaign has **not** started. The required 16-scene sizing gate found incompatible PEC scene/grid combinations and a partial 140 ns resource projection of 18.16 hours for 128 lineages on four GPUs, above the 14-hour data allocation. No v2 CNN has been trained or evaluated on the frozen new test set. See the [implementation plan](IMPLEMENTATION_PLAN.md), [current progress](PROGRESS.md), and [measured pilot contract](docs/v2_pilot.md).

The planned main model is a five-level residual U-Net with widths 64/128/256/512/512 and 26,850,497 parameters. It receives nine 512×512 maps and includes an eight-head bottleneck attention block. The width-16 comparison has 1,683,761 parameters. Both produce positive x/y density profiles projected to exact requested cell counts and an adjacent-cell ratio no greater than 3. The main model's FP16 training-step memory probe selected microbatch 8 at 16.16 GiB on a TITAN RTX.

## Run checks

The active `.venv` and editable package installation point to this directory. From the project root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_local_archive.py
.venv/bin/python scripts/run_mesh_cnn_v2.py launch --output runs_v2/c0_c2_pilot
cat runs_v2/c0_c2_pilot/campaign_status.json
```

The launch command currently stops at `sizing_gate_stopped` before any bulk FDTD or training work. The saved profile is `runs_v2/profile_c0_c2/profile.json`. The archive verifier checks all relocated links and representative file hashes; use `--full` to hash every inventoried archive file.

## Project map

- `src/scattermesh/`: retained TMz solver and versioned C0–C2 geometry, candidate, reference, dataset, CNN, training, and evaluation modules.
- `scripts/run_mesh_cnn_v2.py`: measured sizing gate and deadline-aware campaign entry point.
- `docs/v2_pilot.md`: numerical contract, resource evidence, positive scientific gate, and C3–C9 roadmap.
- `runs_v2/`: new portable outputs, including the 1,104-example qualified historical training import.
- `archive/scattermesh_2026-09-23/`: frozen original scattering source, historical runs, inventories, environment records, and restoration checks.
- `archive/receiver_cnn_2026-09-22/`: preserved receiver-CNN project with 404 repaired relative links.

The project-root `runs/` path is a relative compatibility link into the scattering archive. Historical manifests remain unchanged and may contain former paths as provenance; new import manifests resolve to local archived artifacts. The former Codex worktree is absent. The original project documentation remains in the archives rather than in this active README.
