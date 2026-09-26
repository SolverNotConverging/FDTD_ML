# Mesh-CNN v2 current progress

Updated 23 September 2026. The working root is `/home/s2307298/projects/FDTD_ML`. The [implementation plan](IMPLEMENTATION_PLAN.md) is authoritative; [the C0 protocol](docs/v2_pilot.md) describes the running first experiment.

The user resumed execution. The first 18-lineage, 36-condition C0 calibration finished. Its strict 0.5%/1% spatial gate qualified 6/18 references and 12 teacher searches; the median raw teacher gain on those qualified conditions was 1.315×. An extended 1%/2% gate through 768 cells qualified 14/18 scenes; three lossless high-permittivity cases remained unsettled and one epsilon_r=12 ellipse remained spatially unresolved. Representative dielectric circle, dielectric ellipse, and PEC circle references passed independent duration, quadrature, contour, and PML controls.

A corrected 128-lineage C0 acquisition completed at `runs_v2/c0_restart/acquisition_002`: 96 train, 16 validation, and 16 untouched test lineages were frozen. The qualified dataset contains 262 training examples from 68 lineages and 39 validation examples from 10 lineages, across all three primitive families and both material kinds. The first acquisition manifest was stopped before completion because its held-out material split was imbalanced and is marked aborted under `acquisition_001`. The fresh large CNN is training with saved checkpoints; a separate watcher is queued to evaluate its three best checkpoints on the 39 validation conditions. The test split has not been opened. Current state is in `runs_v2/project_execution_status.json`.

## Execution state

**C0 CNN training is active.** The first four epochs reduced validation profile loss from 0.00276 to 0.00192, and subsequent epochs continue from saved training state. Validation FDTD selection is queued after training finishes; no C0 test condition has been used for model selection.

The revised objective is a fresh C0 start with more data and optimized teachers, progressing toward five silhouette families, multiple material regions, and mixed scenes. Resolution-demand concentration replaces low occupied area as the organizing concept: highly occupied material with localized demanding regions is in scope after numerical qualification.

The old 64/16/24 compact design, ten-epoch C0 warm-up, and 2.92-hour workload estimate do not describe the new curriculum. Existing launch scripts still implement earlier schedules and need migration. Previous planning text is preserved under `archive/plan_revisions/2026-09-23_before_sparse_curriculum_restart/`.

## Implemented foundations

| Capability | Current evidence | Remaining work |
|---|---|---|
| Local project/archive | Local environment and editable install; original scattering/receiver archives; 404 repaired archive links and restoration records | Preserve provenance and verify newly imported artifacts |
| Compiled GPU FDTD | Cython-bound cooperative CUDA loop covers updates, CPML, source, PEC treatment, DFT and checks; zero transfers during stepping | Qualify any new material/background/region formulation in the same backend |
| Geometry | Circles, ellipses, rectangles, polygons, concavity, stars, smooth lobes; separated collections; limited hole/ring support | General touching/nested material partitions and complex feature qualification |
| Mesh output | Positive density profiles; exact cell counts; monotone axes; adjacent ratio at most 3 with recorded repairs | Richer search parameterizations for complex regions and fine features |
| Teacher search | Seeded 12-parameter differential evolution, 96-trial pilots, per-condition worker promotion of three optimized profiles alongside six fixed candidates | Search-budget calibration, failed-trial replay, deadline robustness, reference-aware target qualification |
| Learning | Fresh large C0 residual U-Net training, set-valued distillation, FP16, saved training state; validation FDTD selection queued | Finish C0 training/selection, expanded stage scheduler/replay, updated region channels |
| Silhouettes | Procedural aircraft/ship/vehicle outlines and pilot target generators | Bird, warship-specific, satellite and telescope families; controlled composite materials and held-out cohorts |
| Demand descriptors | Geometric area, bounds and projection metrics exist | Demand maps, entropy/effective-support, spatial/interface and axis-demand diagnostics; validation against teacher gains |
| Full material background | Existing material coefficient machinery supports qualified scatterers in vacuum | Incident source, contrast formulation, PML, normalization, and far-field background generalization |

The main CNN currently has 26,850,497 parameters with nine 512×512 maps; the comparison model has 1,683,761 parameters. Neither has new curriculum training results. The archive import contains 1,104 historical training examples, subject to qualification and split checks for reuse.

## Completed development evidence

- The saved three-repeat paired TITAN RTX benchmark gives end-to-end speedups over legacy Torch of 66.6× for a 96² dielectric circle, 35.8× for a 32² conformal PEC rectangle, and 43.4× for a 64² distributed dielectric scene. Maximum paired complex far-field disagreement is `3.01e-15` relative L2. Records: `runs_v2/cuda_fdtd_benchmark_v1/report.json`.
- The eight-scene compiled-CUDA development profile accepted eight references under its exploratory compact policy and passed representative duration, quadrature, contour, and PML probes. Six fixed teacher policies were weak outside the dielectric circle. Its 2.92-hour estimate includes an extrapolated 96-evaluation search at 144 train/validation conditions, not the expanded curriculum. Records: `runs_v2/compact_poc/profile/compact_profile.json`.
- At 48², 96-trial optimization found raw joint-score ratios uniform/teacher of 1.115 for a development ship, 1.111 for a vehicle, and 1.149 for an analytic-reference PEC circle. The low-contrast C3/C5 scenes and aircraft showed no resolved gain in that search. Ship/vehicle gains persisted at 1.051 and 1.128 against independent 384-cell reference fields. These are development observations, not CNN or frozen C8 results. Records: `runs_v2/teacher_optimization_pilot_v1/` and `runs_v2/teacher_reference_check_v1/summary.json`.
- A lossless epsilon_r=8 two-object development case gave a 5.80× raw-score ratio; its 256→384 complex-field and width differences were 0.409% and 0.361%. This is promising search and spatial-refinement evidence; it does not by itself complete all independent reference checks. A lossless epsilon_r=8 ship gave 4.13× with a still unresolved 1.60% field reference change. Records: `runs_v2/teacher_material_pilot_v1/summary.json`.
- A lossless epsilon_r=12 vehicle had tail ratio `1.91e-5` after 560 ns and did not meet the settling gate. Its separate tan-delta≈0.02 variant settled, but spatial refinement remained unresolved. The conditional loss prototype skips settled lossless cases and adds a new material variant only for an unsettled case. Earlier unrestricted loss sweeps are exploratory artifacts and do not define the current policy. Records: `runs_v2/teacher_material_loss_pilot_v1/conditional_summary.json`.
- One compiled-GPU worker smoke completed six fixed and three optimized candidate slots and built a readable dataset with nine valid targets. Records: `runs_v2/optimized_worker_smoke_v1/`. This establishes one integration path, not full deterministic-resume or new curriculum validation.
- Previous v2 development studies cover limited C3 pairs, C4 gaps, C5 layout/count sweeps, and C6 dielectric rings. They contain both gains and negative outcomes, including weak fixed-policy C5 performance and unresolved cavity references. Their detailed descriptions are retained in the archived progress document and raw `runs_v2/c3_pair_qualification_v1/`, `c4_gap_sweep_v2_near_field/`, `c5_layout_qualification_v2/`, `c5_6to10_qualification_v1/`, and `c6_topology_qualification_v1/` artifacts.

## Verification and current limits

Before optimizer integration the full host-GPU suite passed 114 tests with no skips. The current C0 manifest, worker, teacher, training-schedule, and qualification subset passed 20 tests. The complete expanded project suite has not been rerun since the C0 changes. C0 calibration and acquisition are now real scientific runs; the former planning pause ended at the user's request.

General material-region partitions, a non-vacuum background, entropy-based descriptors, and the five-family curriculum are **planned capabilities**. Current source and far-field normalization use vacuum constants. The new plan requires a consistent background extension before domain-filling materials are claimed.

The GPU host has four TITAN RTX devices. The active environment uses Torch `2.14.0+cu130` for learning and CUDA toolkit 13.2 targeting `sm_75` for the compiled FDTD extension. This tool environment requires host-device access for GPU commands.

## Next work

1. Finish the active large-CNN C0 training. The qualified-label gate passed with 68 training and 10 validation independent lineages.
2. Evaluate the three retained checkpoints on the 39 qualified validation conditions using compiled CUDA FDTD; freeze the selected checkpoint before opening the untouched C0 test split. Produce paired teacher/CNN/uniform accuracy and mesh-budget panels.
3. Assess optimization plateaus at 96/192/384 trials on selected development conditions before committing to larger acquisitions. Add demand descriptors and balanced acquisition for subsequent stages.
4. Expand C0 toward 512 lineages only if learning curves and measured cost justify it. Continue cumulative C1–C5 learning, then qualify internal material partitions and backgrounds as needed before C6–C9 composition experiments.

Every new FDTD campaign uses only the compiled CUDA kernel. Settled lossless high-permittivity cases remain lossless; optional settling-driven loss variants retain separate physics, provenance, baselines, and labels.
