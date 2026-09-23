# Mesh-CNN v2 implementation plan

Updated 23 September 2026. The sole project root is `/home/s2307298/projects/FDTD_ML`; new outputs belong in `runs_v2/`. The original scattering project is preserved under `archive/scattermesh_2026-09-23/`, with the receiver-CNN archive beside it. This document describes the active project only.

## Research objective

Predict nonuniform tensor-product meshes for 2D TMz conformal FDTD scattering. The primary comparisons are **raw complex-field and scattering-width accuracy at the same spatial cell count** and **minimum tested cell count meeting a fixed accuracy target**. Sparse objects should use fewer cells than a uniform grid at equal accuracy. Stable time step, update count, runtime, and memory are measured alongside accuracy. The time-step term in teacher ranking is deliberately weak so it does not drive the CNN toward a uniform grid.

A positive C0–C2 pilot requires a complete valid new-lineage test evaluation. On new dielectric test conditions, at least 75% of cases must improve raw joint accuracy by 5% or more and the median improvement must reach 1.2×. PEC results are reported independently. Missing these provisional gates produces a remediation report; the frozen test set is not retuned.

## Current execution decision

The 16-scene GPU sizing profile has stopped bulk execution. Only nine shape/material rows yielded complete compatible projections. Under the recorded 140 ns assumption, those rows project to 18.16 elapsed hours for 128 lineages or 36.31 hours for 256 lineages on four TITAN RTX GPUs. These are partial projections, not full estimates or rigorous lower bounds. Seven PEC examples had at least one incompatible tested grid; four dielectric examples were unsettled at 192 cells after 70 ns, and five PEC examples exceeded the 50,000-step profiling cap there. The approved data phase allows 14 hours. No new teacher corpus, trained v2 checkpoint, or frozen v2 test result exists. The runner records `sizing_gate_stopped` before launching bulk workers.

The next numerical milestone is to characterize the rejected PEC scene/grid combinations and the cost of settled references, then revise and re-profile the protocol. Preserve exact geometry and explicit split-edge rejection. A larger time allocation alone will not resolve incompatible PEC grids. Freeze any revised scene, grid, and reference rules before generating labels or touching the new test set.

## C0–C2 pilot contract

### Physics and geometry

- Retain the public `scattermesh` simulation interfaces and 2D TMz nonuniform Yee solver on a 1.2 m square. Observe 0.8, 1.0, and 1.2 GHz.
- C0 varies size and position of circles, ellipses, and rectangles. C1 adds rotation and aspect ratio. C2 adds triangles, convex and concave polygons, stars, and smooth-lobed objects, introducing corners, concavity, and changing curvature.
- Use continuous geometry for solver intersections, material sampling, bounds, rasterization, and feature descriptors. Ellipses use analytic conic intersections; polygons use segment intersections; smooth lobes use closed spline boundaries.
- New scenes are approximately 75% dielectric and 25% PEC. Dielectric permittivities are 2, 4, 8, and 12, with lossless and lossy examples. Occupied area is 0.5–8% and projected support is at most 35% on each axis. Qualified historical training data may retain their original broader material range.
- Reject unsupported PEC split edges and unresolved gaps explicitly. Never shift boundaries or close gaps to make a candidate feasible.

### References and teacher data

1. Profile 16 independent scenes spanning all eight shape families and both material types. Size the full corpus at 256 lineages if feasible; otherwise select a complete balanced 128-lineage corpus. If neither fits the 14-hour data allocation, report the estimate before bulk launch. **This gate currently fails.**
2. Split whole geometry lineages approximately 75%/12.5%/12.5% into training, validation, and test, stratified by family and material. Keep related transformations together. Use two incidence angles per scene and exact square budgets of 32, 48, 64, and 96 cells per axis.
3. Use analytic references for circles. Qualify other references by uniform refinement through 192, 256, and 384 cells per axis, escalating to 512 when necessary. Require field tail below `1e-5`, maximum adjacent complex-field variation of 0.5%, and separate duration, material-quadrature, contour, and PML variations at most 0.5%. Store reference uncertainty. Unsettled or incompatible cases cannot become labels.
4. Test up to 12 meshes per condition: uniform, four geometry policies, four seeded smooth-density perturbations, and three local refinements of the best valid first-round mesh. Retain all candidate axes, complex spectra, raw physical errors, stable time step, execution status, and provenance. Keep cases where uniform wins and retain multiple valid targets.
5. Import qualified historical **training** examples only, with original manifests preserved and new portable paths. Draw 25% of training examples from the historical import. The imported set currently contains 1,104 examples; no historical validation or test examples entered it.

Teacher ranking uses the existing normalized complex-field plus 0.25 times floored log-scattering-width loss:

```text
L_accuracy = L_complex + 0.25 L_width
J = L_accuracy * (dt_uniform / dt_candidate)^0.05
```

Re-rank saved candidates at exponents 0, 0.02, 0.05, and 0.1 for sensitivity analysis. Use exponent 0.05 for training targets. Model selection and scientific claims use raw physical accuracy. Numerical fingerprints are separate from scoring and model fingerprints so physics results remain reusable after CNN changes.

### CNN and training

- Main model: five-level residual U-Net with widths 64/128/256/512/512, GroupNorm, SiLU, skip connections, bottleneck conditioning, and one eight-head spatial-attention block at the lowest resolution. It has 26,850,497 parameters.
- Comparison model: the same architecture with base width 16 and 1,683,761 parameters. Initialize both afresh; reuse historical data rather than old checkpoint weights.
- Input: nine 512×512 maps (the seven physical maps plus normalized x/y coordinates) and shape/material/illumination/budget conditioning. Output: positive x/y density profiles projected deterministically to exact requested cell counts, positive widths, and adjacent-cell ratios no greater than 3. Record projection repairs.
- Distill against a set of physics-scored valid profiles. Use AdamW at `3e-4`, weight decay `1e-4`, FP16, and effective batch 32 with gradient accumulation. Select the largest microbatch among 1, 2, 4, and 8 that stays below 20 GiB. The main-model probe selected 8 at 16.16 GiB allocated memory on this server.
- Train up to 120 epochs: 20 on C0, 20 on cumulative C0–C1, then up to 80 on cumulative C0–C2 with patience 20 in the final phase. Evaluate the three best validation-profile checkpoints using validation FDTD, select by raw physical accuracy, and freeze that checkpoint before testing.

## Automated campaign and deliverables

The planned overall limit is 24 hours: up to 1 hour for profiling and sizing, 14 for references and teachers, 6 for training and validation selection, and 3 for frozen testing and reporting. Four GPUs run independent FDTD workers; the two CNNs train on separate GPUs. The runner saves restartable state, enforces phase deadlines, and marks missing work incomplete. Incomplete, unconverged, and numerically incompatible cases cannot become labels. Campaign completion and positive scientific evidence are distinct outcomes.

Required output includes raw complex-field and width error versus cell count; same-budget improvement by shape and material; minimum tested cell counts meeting both 2%, 5%, and 10% error limits, with unmet limits identified; cell-saving ratios and reference uncertainty; time step, update count, runtime, and memory; large-versus-small comparison and penalty sensitivity; representative mesh and scattering curves including failures; and counts of incompatible, unconverged, budget-limited, and incomplete cases.

The v2 runner is [`scripts/run_mesh_cnn_v2.py`](scripts/run_mesh_cnn_v2.py). The current machine-readable gate is [`runs_v2/c0_c2_pilot/campaign_status.json`](runs_v2/c0_c2_pilot/campaign_status.json). Detailed pilot assumptions and measured evidence are in [`docs/v2_pilot.md`](docs/v2_pilot.md). The prior implementation plan is preserved verbatim in the scattering source archive.

## Later curriculum

- C3–C5: mixed pairs, controlled gap sweeps, and 3–10 objects; measure spatial spread as well as object count.
- C6–C7: holes, open cavities, thin sections, and multiscale details after numerical qualification.
- C8–C9: frozen unseen engineering silhouettes and distributed scenes.
- Before manuscript claims: independent solver comparison, repeated training seeds, larger frozen tests, and confidence intervals.
