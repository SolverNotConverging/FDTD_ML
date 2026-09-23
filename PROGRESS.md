# Mesh-CNN v2 project progress

Updated 23 September 2026 after the project-wide proof-of-concept revision. This report covers the active project in `/home/s2307298/projects/FDTD_ML` on `codex/mesh-cnn-v2`. The [implementation plan](IMPLEMENTATION_PLAN.md) defines the entire C0–C9 roadmap; [the pilot protocol](docs/v2_pilot.md) specifies its first demonstration.

## Current status

**The current objective is a compact demonstration on unseen engineering-like silhouettes and distributed sparse scenes. The revised executable workflow is not yet implemented.** C0–C2 and ordinary C3/C5 scenes provide training preparation; bounded C8/C9 scenes belong in the first evaluation. Extreme gaps, topology, very thin details, and larger scenes follow the stage instructions in the project-wide plan.

No v2 CNN has been trained. No new teacher corpus, frozen target-scene test manifest, or C8/C9 physical evaluation has been completed. The 1,104-example historical import is available training data, not a new target-scene dataset.

The existing runner still implements the superseded single-object C0–C2 design: fixed 128/256 lineage choices, a 14-hour data gate, twelve teacher candidates per condition, extensive per-condition reference probes, and mandatory large/small model selection. Its recorded `sizing_gate_stopped` result is retained as evidence about that design. Removing those requirements from the documentation has not changed the Python implementation.

## Capability status across C0–C9

| Stage or capability | Implemented evidence | Work remaining |
|---|---|---|
| Numerical core and C0/C1 | TMz CPU/CUDA solver, circles/ellipses/rectangles and rotations, continuous intersections/material sampling, analytic circle checks | Broader development calibration and trained v2 results |
| C2 geometry | Triangles, convex/concave polygons, stars, smooth lobes; nine-map single-object inputs | Broader procedural boundary diversity and useful teacher demonstrations |
| C3 pairs / C5 groups | Reusable solver and raster components exist; v2 scene construction still returns one object | Collection interface, proximity/conditioning, per-object candidates, scene metrics, qualification |
| C4 small gaps | Unsupported PEC split edges are explicitly rejected | Supported-range mapping; any required solver changes before labels |
| C6 topology / C7 multiscale | Current single-ring geometry and input raster impose limits | Rings/voids/cavities, thin-feature qualification, representation and reference checks |
| C8 silhouettes / C9 distributed tests | Project-wide stage instructions and compact evaluation scope are now documented | Definitions/import, grouped provenance, frozen target suite, physical evaluation |
| CNN and learning | Large and small networks, projection, distillation, resume and validation/test framework | Scene-aware adaptation, cumulative multi-object schedule, heuristic comparison; no trained v2 checkpoint |
| Compact campaign | Revised workload and evidence protocol documented | Configurable counts/conditions, adaptive references, measured costs, updated orchestration |

## Reusable completed foundations

- The original scattering source is preserved at commit `c514697d7134e41389098f3c28ef847d9f3bc203` on `codex/archive-scattermesh-20260923`. Its source, datasets, spectra, checkpoints, logs, paused states, and 59,178 run files are in `archive/scattermesh_2026-09-23/`. The receiver archive is adjacent. All 404 receiver links resolve directly inside that archive; original inventories and historical manifests remain unchanged.
- The active environment resolves this project locally, uses the `scattermesh` prompt, and no longer advertises the old receiver package's editable installation. Dependency and relocation records are preserved in archive metadata. The former Codex worktree is absent; the project `.codex` entry is an empty read-only mount using 0 bytes.
- Historical root scripts, configurations, reports, learning modules, and tests have been removed after archive verification. The active tree retains the reused solver, numerical regressions, and v2 code. New outputs use `runs_v2/`; `runs/` remains a local archive compatibility link.
- Implemented exact-budget density projection with grading at most 3, weak time-step teacher scoring, set-valued profile distillation, and restartable FDTD records with raw spectra, axes, provenance, status, stable time step, runtime, and CUDA memory.
- Imported 1,104 accepted historical training examples with portable archive paths and teacher scores recalculated at exponent 0.05. No historical validation or test entries entered that import.
- Implemented the 26,850,497-parameter residual U-Net and 1,683,761-parameter comparison model, FP16 memory probing, saved optimizer/RNG state, validation FDTD selection, and frozen-checkpoint reporting for the earlier single-object protocol. These provide code foundations; they do not establish the revised scene capabilities.

## Measured evidence and its limits

- Four TITAN RTX GPUs were available during the earlier profile. Its 16 shape/material rows ran in 109 seconds; only nine yielded complete compatible projections, comprising eight dielectric and one PEC row. Seven PEC examples had an incompatible tested grid. At 192 cells, four dielectric examples were unsettled after 70 ns and five PEC examples exceeded the 50,000-step profiling cap.
- Under that profile's 140 ns assumption, compatible rows projected to 18.16 elapsed hours for 128 lineages or 36.31 hours for 256 on four GPUs. These are partial projections for the superseded campaign, not complete estimates, rigorous lower bounds, or estimates for the new compact design. See `runs_v2/profile_c0_c2/profile.json` and `runs_v2/c0_c2_pilot/campaign_status.json`.
- A dielectric-circle reference smoke under the earlier strict protocol reached 512 cells per axis, with separate duration, quadrature, contour, and PML checks and maximum variation 0.1853%. This does not establish a universal reference-grid requirement. At 32 cells, star dielectric and smooth-lobed PEC CPU/CUDA far fields agreed within `5e-16` relative L2; backend agreement alone is not physical convergence.
- The main CNN's FP16 training-step probe admitted microbatches 1, 2, 4, and 8 below 20 GiB. Microbatch 8 peaked at 16.16 GiB. This establishes memory fit for that encoding, not trained quality or a campaign runtime.
- The most recent code verification before this documentation revision was **52 tests passed, 6 skipped**, with Ruff checks passing. All 404 archive links and representative file hashes passed verification; full receiver and scattering inventories had also been audited. Representative restoration checks are recorded in `archive/scattermesh_2026-09-23/metadata/RESTORATION_VERIFICATION.json`.

## Next work

1. Generalize v2 scenes and inputs to multiple continuous objects, add silhouette definitions and lineage, and preserve numerical regressions. Follow the C0–C3/C5 instructions in the project-wide plan.
2. Make experiment counts, split conditions, teacher policies, reference qualification, and deadlines configurable. Add a fixed geometry-based baseline and reference-sensitivity reporting. The current CLI is not the revised campaign launcher yet.
3. Qualify six to eight development cases spanning primitives, silhouettes, and separated objects. Measure available mesh benefit and cost; determine numerical tolerances and freeze the compact protocol before bulk execution.
4. Start from roughly 64 new training lineages, 16 validation lineages, and 24 frozen target scenes: 12 silhouettes and 12 distributed scenes. Use up to six teachers at one angle and two sampled budgets per training lineage. Preserve a 24-hour campaign ceiling with phase allocations based on new measurements. No measured completion estimate is available yet.
5. Train the large CNN with cumulative scene coverage, select with validation FDTD, freeze, then evaluate uniform/heuristic/CNN on C8/C9 target conditions. Report the full suite, including limits and failures. The small-model comparison follows the main result if resources allow.
6. Extend C4/C6/C7 and larger C5/C9 regimes using the per-stage qualification, replay, provenance, and frozen-evaluation rules. Subsequent studies strengthen the evidence without turning the first demonstration into a large solver-validation campaign.
