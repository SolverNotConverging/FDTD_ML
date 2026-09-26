# First resumed experiment: C0 optimized-teacher calibration

Revised 23 September 2026. C0 calibration and the first corrected acquisition batch are complete under `runs_v2/c0_restart/acquisition_002`; large-CNN training is active. This file specifies the first experiment in the [C0–C9 resolution-demand curriculum](../IMPLEMENTATION_PLAN.md). The former compact C8/C9 pilot protocol is preserved in `archive/plan_revisions/2026-09-23_before_sparse_curriculum_restart/docs/v2_pilot.md`.

## Purpose

Establish that optimized teachers provide useful, numerically resolved mesh targets across shape, position, scale, and material, then train the first C0 CNN. The result must supply manuscript evidence for the learning mechanism and establish realistic costs for the expanded curriculum.

Use the existing 2D TMz physics, 1.2 m square domain, 0.8/1.0/1.2 GHz observations, and compiled CUDA time loop. All new FDTD work uses the compiled kernel; CPU/Torch solvers are only numerical regression references.

## Step 1: balanced calibration

The first frozen calibration manifest has 18 independent circles, axis-aligned ellipses, and rectangles and 36 budget/angle conditions. It spans varied positions and scales, lossless epsilon_r 2/4/8/12, one prescribed lossy material class, and PEC. Inspect its actual area and projection distributions before claiming that all proposed strata have been covered. This panel is developmental and never part of final test claims.

Record union area, x/y projection coverage, physical/electrical feature sizes, and clearances. Occupied-area bins describe the first free-space primitive panel; they are not the project's definition of sparsity or an exclusion rule for dense material. Develop resolution-demand maps and entropy/effective-support descriptors, with spatial/interface and x/y demand diagnostics, as specified in the main plan. Correlate them with measured teacher gains before using them as acquisition gates. Include uniform-demand controls. Dense heterogeneous hosts follow region-partition qualification, and true non-vacuum domain backgrounds require consistent source/PML/far-field support.

Sample the legal domain interior. Respect source/monitor/PML constraints without moving geometry differently for different meshes.

Use analytic circle references after solver checks. For other shapes, compare successive uniform resolutions and probe independent duration, quadrature, contour, and PML sensitivities. Start with a target of at most 0.5% complex-field and 1% width variation for reference refinement; freeze actual tolerances from calibration and the gains to be resolved. These empirical differences are not certified error bounds.

The initial settling ladder is 70/140/560 ns with finite fields and tail below `1e-5`. A longer ceiling may be proposed during calibration but must be frozen before acquisition. Separate settling, spatial nonconvergence, geometry incompatibility, and budget exhaustion.

Keep settled lossless high-epsilon conditions. A failing candidate is rejected without changing material. If the original physical condition remains unsettled at the declared maximum duration, preserve its failure and optionally define a separate lossy variant with the same geometry ancestry and split. Record sigma, loss tangent at its reference frequency, and why the variant exists. Recompute its reference and baselines. Prescribed loss is allowed as its own physical input and is not this convergence fallback.

## Step 2: teacher-search calibration

Optimize each complete geometry/material/illumination/budget condition independently. Begin with the current 12-parameter positive smooth-density search, seeded by uniform and existing geometry policies. Add material-aware seed information as implementation work.

Compare search budgets of 96, 192, and 384 FDTD evaluations on the development panel. Measure raw accuracy improvement, score improvement, rejection rate, target diversity, and wall time versus evaluation count. Use multiple seeds on a representative subset to estimate search variability. The output is the best mesh found within the tested search, not a certified global optimum.

Project every trial to exact cell counts, positive widths, and adjacent ratios at most 3. Preserve continuous boundaries and record repairs. Score accepted fields with:

```text
L_accuracy = L_complex + 0.25 L_width
J = L_accuracy * (dt_uniform / dt_candidate)^0.05
```

Save all axes, raw spectra, physical errors, dt, attempts, costs, statuses, and provenance. Keep distinct good alternatives and uniform controls. Re-score dt exponents 0/0.02/0.05/0.1 from saved results. Resolve apparent gains against available reference variants; exclude unsettled, incomplete, incompatible, or unresolved labels.

Before unattended acquisition, verify optimizer resume, including failed trials, deadline interruption, target promotion, and dataset masks. The existing worker smoke verifies one ordinary condition only.

## Step 3: C0 acquisition and training

The proposed C0 target is 512 independent geometry lineages: approximately 384 training, 64 validation, and 64 test. Acquire balanced 128–256-lineage batches, subject to measured cost. Keep all transformations and material descendants within one split.

Start with 4–8 sampled physical conditions per train/validation lineage, balanced over material, two incidence angles, and four cell budgets. Full controlled sweeps belong to separate development panels. Every sampled train/validation condition receives teacher optimization. With 448 train/validation lineages, this means about 1,792–3,584 optimized conditions before any extra development work; it is substantially larger than the old compact experiment and needs a new measured forecast.

Train the existing large residual U-Net from fresh weights. Start with AdamW at `3e-4`, weight decay `1e-4`, FP16, and effective batch 32; recheck microbatch memory. Propose up to 80 C0 epochs with validation patience 15, finalized after learning-curve calibration. Historical data can be a qualified subset of training draws, with no historical validation/test leakage.

Use profile validation followed by validation FDTD for the three best checkpoints, freeze one, then open the C0 test. C1 and later stages continue cumulatively from the selected checkpoint with balanced replay. The expanded scheduler remains planned work; current launch commands implement the superseded schedule.

## Required C0 results

- Learning curves and optimization progress versus FDTD calls.
- Uniform, fixed heuristic, optimized teacher, and CNN meshes/fields on paired shape, material, position, and size examples.
- Raw complex-field and width errors versus cell budget; minimum tested counts meeting both 2%, 5%, and 10% targets, including unmet thresholds.
- CNN-to-teacher accuracy gap, uniform-winning cases, uncertainty-sensitive rankings, and all failure statuses.
- Cell savings versus demand concentration/entropy, absolute demand, interface/feature scales, occupied area and x/y coverage; dt, update count, runtime, memory, teacher-search cost, and CNN inference cost.
- A frozen validation panel to track forgetting through C1/C2 and a separate untouched final silhouette/scene design.

Before scaling acquisition, require usable qualified teachers across the declared strata, teacher gains that survive reference variation where gains are claimed, and a feasible measured batch forecast. Weak teacher gains call for improving search or revisiting the physical regime; they do not justify deleting uniform-winning conditions.

## Execution state

The user resumed execution on 23 September 2026. The first calibration qualified 6/18 references at a 0.5%/1% spatial threshold. The extended 1%/2% policy through 768 cells qualified 14/18; three scenes were unsettled and one remained spatially unresolved. Independent duration, quadrature, contour, and PML controls passed for representative dielectric circle, dielectric ellipse, and PEC circle references. Four compiled CUDA workers completed the corrected 128-lineage batch. It yielded 68 qualified training and 10 qualified validation lineages, with 262 and 39 examples respectively; the 16 test lineages remain untouched. The first split manifest was aborted because validation/test materials were imbalanced and contributed no CNN labels. Fresh large-CNN training is active with saved checkpoints. A validation watcher will evaluate the three best checkpoints on the 39 qualified validation conditions, then freeze one only if all three evaluations are complete. The old 2.92-hour compact forecast does not apply to this protocol.
