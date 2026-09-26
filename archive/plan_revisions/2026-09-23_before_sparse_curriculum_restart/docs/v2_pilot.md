# Mesh-CNN v2: first demonstration protocol

Updated 23 September 2026. This document specifies the compact first experiment within the [project-wide C0–C9 implementation plan](../IMPLEMENTATION_PLAN.md). Later stages have their own instructions there. The scene, split, teacher-worker, adaptive-reference, training, frozen-evaluation, and phase-deadline `launch-compact` interfaces exist. Bounded CPU C3 pair and C5 matched-area layout studies provide development evidence; the C5 fixed-policy results are weak on sparse layouts. GPU calibration, v2 CNN training, and measured C8/C9 target-scene results remain outstanding.

## Purpose and scope

Demonstrate learned nonuniform meshing on held-out individual silhouettes and distributed scenes with actual 2D TMz conformal FDTD results. Use C0–C2 foundation geometries and ordinary C3/C5 separated objects for training. Include bounded C8/C9 tests in this first experiment. Subsequent C4/C6/C7 work expands small-gap, topology, and multiscale capability.

Use simplified aircraft-like, ship-like, and vehicle-like outlines at modest electrical sizes. The initial physical setting is the existing 1.2 m square with 0.8/1.0/1.2 GHz observations. The core material study starts with lossless dielectric relative permittivities 2 and 4. Qualified historical data can retain their recorded broader material range. Lossy and PEC cases are separately declared extensions or supplements; a fixed 25% PEC quota is not required.

Define resolvable feature, gap, material, and boundary-clearance ranges during development calibration. Continuous geometry is authoritative. Preserve explicit rejection of incompatible PEC split edges and keep the geometry fixed across mesh methods. Freeze the scope and list every requested final condition before evaluation.

## Development and frozen splits

First use six to eight development cases spanning primitives, silhouettes, and separated objects. Measure whether nonuniform teacher meshes offer gains, whether the numerical references can resolve those gains, and what the full simulation costs are. Development silhouettes have different source templates from the target tests.

The starting corpus design is:

| Set | Starting size | Conditions and use |
|---|---|---|
| Historical training | 1,104 imported examples | Start with 25% of training draws; training provenance only |
| New training | Approximately 64 lineages | One sampled incidence angle and two sampled budgets per lineage; up to six teachers per condition |
| Validation | Approximately 16 separate lineages | Compact conditions fixed before training; profile validation and FDTD checkpoint selection |
| Target silhouettes | 12 scenes | Held-out templates; four budgets and two incidence angles |
| Target distributed scenes | 12 scenes, initially 2–6 objects | Held-out templates/layouts; four budgets and two incidence angles |

Use 32/48/64/96 cells per axis. Distribute sampled training budgets and angles across the intended conditioning range. The starting teacher allocation is at most 768 new training candidate simulations; reference, calibration, validation, and test work is additional. Select the validation condition count using measured costs before freezing the campaign.

Group all transformations of a source shape and all related layout derivatives within a split. Reused test templates across scenes remain statistically grouped. Record unique template/lineage counts as well as scene counts. Final test geometry may be defined in advance, but test physics, teacher rankings, and model results must not guide training or protocol selection. State exactly whether the held-out distinction concerns templates, construction families, arrangements, or a combination.

## Scene representation and metrics

The code now accepts legacy single-object scenes and a versioned object-collection schema with per-object materials and IDs. Collections feed solver material/intersection queries, rasterization, candidate generation, and whole-scene monitor enclosure. The initial C3/C5/C8/C9 path requires positive axis-aligned bounding-box gaps; overlaps, tangencies, and split-edge PEC configurations are not silently repaired.

The current nine maps and conditioning include union geometry/material fields, signed distance, boundary proximity, inter-object proximity, x/y coordinates, object count, aggregate material values, feature size, and support metrics. Polygon boundary coordinates and exact conic extrema inform the fixed interface policy. Recheck memory before training because collection maps now use scene-wide rasterization.

For each scene record:

- Occupied union area, divided by domain area.
- Union length of object projections on x and on y, divided by the corresponding domain extent.
- Overall bounding-box extent, minimum separation, smallest represented feature, and object count.
- Material composition, electrical size, and clearances to the PML and observation contour.

Union projection coverage and bounding-box extent are distinct. A scattered collection can have a large overall extent but gaps in its axis projections. Compare compact, aligned, and dispersed layouts at similar occupied area; tensor-product refinement affects whole grid lines, so area sparsity alone does not establish a cell-saving advantage. Replace the single-object 35% support cap with declared scene ranges suitable for these comparisons.

## Teachers and comparison methods

Use up to six candidate meshes per training condition: uniform, three geometry policies, and two deterministic seeded perturbations. Policies must account for all objects and actual boundaries/features. Retain every valid profile and its physics score, rather than saving only the winning mesh. Extra local search is a bounded development diagnostic, not a default requirement on every scene.

Keep the scoring contract:

```text
L_accuracy = L_complex + 0.25 L_width
J = L_accuracy * (dt_uniform / dt_candidate)^0.05
```

Here `L_complex` is the existing normalized complex-field term and `L_width` the existing floored log-scattering-width term. The uniform time step comes from the same scene and spatial budget. Candidate axes, raw spectra, errors, settling status, projection repairs, time step, runtime, memory, and provenance are retained. Exponents 0, 0.02, 0.05, and 0.1 can be compared by re-scoring saved results.

The final comparison is uniform versus a fixed geometry-based interface policy versus the frozen CNN. The fixed policy is implemented and targets actual polygon vertices and conic support coordinates from every object. Its scale and weight still need development calibration and must be frozen before test evaluation. Choosing the best heuristic separately using each test reference would be an oracle search; label such an additional diagnostic explicitly and do not present it as a fixed mesh generator. Final tests need no full teacher search.

Exact spatial budgets, domain, geometry, material, incidence, frequencies, and reference fields must match across methods. Report direct CNN meshes, deterministic grading repairs, and any fallback separately. Uniform-winning and failed predictions remain visible.

## Calibrated reference protocol

The goal is to distinguish physical mesh improvements from numerical uncertainty at affordable cost. The earlier mandatory per-condition 192/256/384/512 sweep and four independent probes remain in the legacy path. The compact profiler tries successive 96/128/192/256 uniform grids and representative sensitivity probes. Compact manifests freeze that policy, and compact teacher/evaluation qualification now uses it. The profiler and adaptive path have not been run on GPU or calibrated.

1. **Calibration set:** check circle results against analytic fields and use representative polygon, silhouette, and multi-object development scenes to assess spatial resolution, simulation duration, PML thickness, contour placement, and material quadrature independently. Declare the supported geometry/material range.
2. **Routine references:** use the analytic solution for supported isolated circles after calibration. For other shapes, compare successive uniform resolutions, starting at 128/192 and escalating to 256 or higher when necessary. A selected resolution must be justified by measured variation; none of these levels is automatically a reference.
3. **Settling and duration:** retain the existing field-tail threshold `1e-5` initially. Verify duration sensitivity for calibrated regimes and repeat it for long-lived or suspicious cases. A small field tail does not prove spatial accuracy, and backend agreement does not prove convergence.
4. **Teacher uncertainty:** evaluate candidate scores against the available reference variants. Preserve alternatives whose ordering cannot be resolved. Require additional refinement or exclude a label when the evidence cannot support its physical ranking; uniform remains a valid outcome.
5. **Final conclusions:** recompute reported errors using the accepted reference variants. Count an improvement or threshold crossing as resolved only when the conclusion survives those variations under the frozen protocol. Otherwise report an uncertainty range or an unresolved result and escalate selectively if the budget allows.

Before bulk launch, store numeric tolerances for spatial/duration/probe variation, duration and step limits, escalation/cost caps, and the rule for unresolved comparisons. The tolerance selection uses development evidence and the effects the experiment aims to measure. Reference variation is an empirical uncertainty estimate, not a certified error bound. Do not loosen thresholds after seeing target results.

Cases outside the calibrated range require fresh qualification or an explicit out-of-scope status. Do not silently replace difficult frozen test scenes with easier ones. Retain numerical fingerprints independently from model and scoring versions so changing the CNN or time-step exponent does not unnecessarily discard physical results.

## Model and selection

Retain the current large residual U-Net with widths 64/128/256/512/512, GroupNorm, SiLU, bottleneck conditioning, and eight-head attention. It currently has 26,850,497 parameters. Nine 512×512 raster inputs describe the geometry; FDTD budget and raster resolution are independent. Output densities are projected to exact cell counts, positive spacings, and adjacent ratios no greater than 3, with repair fractions recorded.

Use set-valued profile distillation, AdamW at `3e-4`, weight decay `1e-4`, FP16, and effective batch 32. The existing memory probe selected microbatch 8 at 16.16 GiB for the current encoding; remeasure if inputs change. Start with 10 foundation warm-up epochs and up to 50 cumulative epochs including ordinary multi-object scenes, with patience 10 in the cumulative phase. Finalize this schedule from development profiling.

Evaluate the three best profile checkpoints using the frozen validation FDTD conditions and raw physical accuracy. Freeze the selected checkpoint and heuristic before running target evaluations. The 1,683,761-parameter small model is a later ablation under the same data and split rules. A single training seed supports a bounded initial demonstration; repeated seeds strengthen later claims.

## Execution and reporting

All new FDTD work in this protocol uses the compiled CUDA backend described in the [project plan](../IMPLEMENTATION_PLAN.md). The Cython binding starts one cooperative device kernel for the complete time loop; CPU and Torch FDTD paths are limited to parity tests and the preserved benchmark. New labels and frozen target results cannot come from them. Build and source-hash validation are required before GPU profiling or a campaign launch.

Retain a 24-hour upper limit for the first revised automated campaign once implementation and calibration are ready. Profile the full proposed workload and reserve time for target evaluation and reporting. Allocate phase deadlines from measured costs, replacing the old fixed 14-hour data gate. Record calibration cost separately. Four GPUs can run independent FDTD jobs; the main CNN trains after data readiness. Further stage campaigns receive their own explicit finite budgets.

If the proposed size does not fit, reduce new training/search work before freezing the workload while preserving both target cohorts. Save optimizer/RNG state and incomplete case records at deadlines. Unconverged and incomplete outputs cannot become training labels. No revised runtime projection or completion promise is available yet.

Required outputs are:

- Raw complex-field and scattering-width error versus cells for all three methods, with both target cohorts reported separately.
- Minimum tested total cells satisfying both 5% and 10% error targets, and optionally 2% where reference sensitivity permits. Use `Nx × Ny` for savings; mark unmet or unresolved targets explicitly.
- Same-budget improvement, error/threshold sensitivity to reference variants, valid coverage, and results grouped by source lineage, material, and spatial arrangement. Do not treat angles, budgets, or related layouts as independent shape samples.
- Representative meshes and scattering curves, including weak/negative results and failures; savings versus union axis coverage and scene spread.
- Time step, update count, runtime, memory, and inference time. Report training and teacher-search cost separately from amortized mesh generation.
- Counts for incompatible, unsettled, reference-unresolved, budget-limited, and incomplete cases; PEC supplement and model-size ablation when available.

A successful first demonstration shows repeatable, reference-resolved gains on held-out examples in both the silhouette and distributed cohorts, with the full frozen suite disclosed. Report medians, coverage, and individual outcomes without a universal success claim from selected examples. The former 75%/1.2× C0–C2 gate is superseded. If the CNN matches the fixed heuristic without improving physical accuracy, frame the result as learning to automate useful mesh construction and quantify any generation-cost advantage. Broader accuracy claims require supporting evidence.

## Implementation checklist

- [x] Multiple-object schema and single-object legacy adapter; shared continuous geometry across supported separated-object paths.
- [x] Scene raster/conditioning, boundary-aware candidates, support metrics, and whole-scene monitor checks.
- [x] Compact grouped splits, per-lineage train conditions, candidate sets, resumable condition workers, test-label exclusion, and phase-deadline `launch-compact` orchestration.
- [ ] GPU-validated calibration of the development reference profiler, ambiguous-teacher handling, and final reference-sensitivity reporting.
- [x] Fixed interface heuristic alongside uniform and CNN; optional small-model orchestration remains later work.
- [ ] Development-case results and measured compact workload; frozen numerical protocol and target manifest.
- [x] Cumulative multi-object training schedule, validation FDTD selection, and post-freeze target evaluation interfaces; no CNN has yet been trained.
- [ ] Complete report with uncertainty, coverage, failures, and provenance.

The current `scripts/run_mesh_cnn_v2.py launch` still executes the superseded C0–C2 logic; use `profile-compact`, `freeze-compact`, and `launch-compact` for the revised path. Compact manifest and launch remain gated on a complete measured profile. No compact GPU profile, trained CNN, or C8/C9 FDTD result exists yet. Use [PROGRESS.md](../PROGRESS.md) for implementation state and the [project-wide plan](../IMPLEMENTATION_PLAN.md) for later-stage work. Existing evidence under `runs_v2/profile_c0_c2/` and `runs_v2/c0_c2_pilot/` remains provenance for the earlier design.
