# Mesh-CNN v2 project-wide implementation plan

Updated 23 September 2026 after the proof-of-concept scope review. This is the current research and implementation plan for the entire C0–C9 project, including the first demonstration and subsequent extensions. The sole project root is `/home/s2307298/projects/FDTD_ML`; new outputs belong under `runs_v2/`. Original scattering and receiver projects remain in `archive/`.

## Objective and claim

Demonstrate that a CNN trained on simpler geometries can predict useful nonuniform tensor-product meshes for **unseen 2D engineering-like silhouettes and distributed sparse scenes**. The evidence must include actual conformal FDTD scattering results on those target scenes in the first completed demonstration.

The primary outcomes are complex-field and scattering-width accuracy at the same spatial cell count, and fewer cells at comparable accuracy. Stable time step, update count, runtime, memory, and mesh-generation time accompany those outcomes. The weak time-step penalty must not dominate accuracy or reward collapse to a uniform mesh.

Use simplified outlines at modest electrical sizes. Aircraft-like, ship-like, and vehicle-like shapes test geometry transfer; a 2D silhouette experiment does not establish full-scale 3D engineering scattering performance. Report the demonstrated range of shapes, materials, electrical sizes, and spatial arrangements.

## Current implementation and next milestone

The repository now has versioned separated-object scenes, scene-wide rasters and material conditioning, polygon/conic boundary-aware mesh policies, C3/C5 generators, limited C4/C5/C6 numerical qualifications, procedural C8/C9 test scenes, and a grouped 64/16/24 suite. A compiled cooperative CUDA kernel performs the complete FDTD time loop with no host/device transfers during stepping. Eight GPU development references and their independent probes are qualified. The compact data worker now runs six fixed candidates plus a bounded smooth-density optimizer for each train/validation condition, retaining three optimized profiles. A one-condition GPU worker smoke produced a readable nine-candidate CNN dataset. The large CNN and portable import of 1,104 historical training examples are available. No v2 CNN has been trained and no frozen C8/C9 target evaluation has run.

A bounded CPU C3 study now qualifies four equal-material dielectric circle pairs varying ordinary separation, a 2:1 size ratio, and diagonal placement. Their references selected 256 cells/axis in three cases and 384 in the size-ratio case; all four passed duration, quadrature, contour, and PML probes. All 80 uniform/interface/center/hybrid mesh evaluations were accepted. The center policy had lower raw joint scattering loss than uniform in 19 of 20 matched-budget conditions, and remained the teacher-score winner under time-step exponents 0, 0.02, 0.05, and 0.1 in those same 19 conditions. Some cases met 5% error at fewer cells, but savings were not universal. This is limited fixed-policy CPU evidence: it does not establish mixed-material or general shape-pair behavior, actual CUDA equivalence, or CNN transfer. See `PROGRESS.md` and `runs_v2/c3_pair_qualification_v1/`.

A bounded CPU C5 study now covers three- and five-object scenes in compact, aligned, and dispersed arrangements using mixed shape families. Each scene has the same 0.0068 m² dielectric area; each count group reuses identical object geometry across its three layouts. All six references selected 256 cells/axis and passed four sensitivity probes. The corrected wider-monitor protocol accepted 158 of 168 mesh evaluations; ten low-budget cases remain unsettled. The nonuniform policies beat uniform raw joint accuracy in only 10 of 24 matched-budget conditions. None reduced cells at 5% joint field-and-width error across this suite; one five-object compact case saved 43.8% at 10%. This is negative evidence for the current fixed policies on sparse collections and a reason to measure the CNN directly. The campaign uses one dielectric permittivity and one incidence angle; it does not complete C5's broader material/count gate. See `PROGRESS.md` and `runs_v2/c5_layout_qualification_v2/` (the narrower-monitor diagnostic run is retained under `v1`).

The C5 lineage generator now optionally emits 2–10 object scenes and per-object `εr=2/4` mixtures while preserving the existing compact-campaign default. Uniform 64-cell CPU compatibility runs for six, eight, and ten mixed-material scenes all settled with the scene-wide monitor; those initial smokes remain recorded in `runs_v2/c5_6to10_compatibility_v1/`.

A subsequent controlled CPU qualification covers six, eight, and ten objects in compact, aligned, and dispersed layouts at matched 0.0068 m² occupied area. All nine homogeneous `εr=4` dielectric references passed the successive-grid, duration, quadrature, contour, and PML checks; eight selected 256 cells/axis and one selected 384. Maximum recorded reference uncertainty was 0.415%, and maximum independent quadrature and contour changes were 0.332%/0.267% and 0.057%/0.060% for complex field/width. Of 81 uniform/center/hybrid mesh runs at 48/64/96 cells, 76 settled, four low-budget aligned cases were unsettled, and one 6-object aligned hybrid projection was numerically incompatible. Uniform had the lowest raw joint scattering loss in every accepted matched-budget comparison and remained the teacher-score winner at time-step exponents 0, 0.02, 0.05, and 0.1. No nonuniform policy reduced tested cells at 2%, 5%, or 10% joint error. This is qualified reference and candidate evidence for one lossless material/frequency/angle slice, not learned-model evidence or broad C5 completion. See `runs_v2/c5_6to10_qualification_v1/c5_layout_qualification_report.json`.

A bounded CPU C4 study covers two 60 mm-radius dielectric circles at four fixed gaps from 2 to 20 mm. All four references passed successive-grid far-field and five-point gap-line checks at 256 cells/axis, plus duration, quadrature, contour, and PML probes. Across 100 uniform/geometry-policy evaluations, center- and hybrid-focused meshes reduced the minimum tested count by 43.8% at 2% field-and-width error and by 55.6% at 5% across this small sweep. This is limited numerical and fixed-policy evidence, not learned transfer: the C3 and C4 samples share narrow material and shape coverage, and no C4 CNN training has run. See `PROGRESS.md` and the raw report in `runs_v2/c4_gap_sweep_v2_near_field/`.

The existing `launch` command still executes the superseded 128/256-lineage C0–C2 design. Use `profile-compact`, `freeze-compact`, and `launch-compact` for the revised path. The refreshed eight-scene GPU profile estimates 2.92 elapsed hours on four GPUs, including 96 optimization trials for each of 144 train/validation conditions; this is a measured-cost projection, not a completed campaign. No compact manifest or CNN training run has been launched. The saved `sizing_gate_stopped` result describes the old design only.

The next milestone is to freeze the optimized-teacher protocol and train the first CNN. The original six fixed policies rarely beat uniform outside the circle. A 96-trial development search improved a lossless `εr=8` two-object scene by 5.80× at 48² with a 0.409% successive-grid complex-field change, and an analytic-reference PEC circle by 1.149×. Development ship and vehicle gains of 1.051× and 1.128× persisted against independent 384-cell reference fields. These observations show teacher-search headroom, not learned transfer. A high-contrast ship reference remains unresolved and a lossless `εr=12` vehicle did not settle at the allowed 560 ns, so neither is a training label in that state. Final test templates remain untouched.

## Project curriculum and milestone order

| Role | Geometry scope | Evidence or purpose |
|---|---|---|
| C0–C2 foundation | Circles, ellipses, rectangles, triangles, convex/concave polygons, stars, smooth lobes; size, position, rotation, and shape variation | Learn geometry-to-density mapping using reused and new data |
| C3/C5 bridge | Ordinary separated pairs and 2–5 mixed objects, with varied positions, orientations, and sizes | Learn multiple-object inputs and distributed allocation |
| C8 demonstration | Held-out aircraft-like, ship-like, and vehicle-like outlines with resolvable features | Measure transfer to unseen silhouette templates |
| C9 demonstration | Held-out distributed scenes, initially 2–6 objects | Measure cell savings as occupied area and axis coverage change |
| Later extensions | C4 extreme gaps; C6 holes/cavities; C7 very thin or multiscale details; larger C5/C9 scenes | Extend the demonstrated regime after numerical qualification |

C8–C9 belong in the first experiment. Completing every intervening difficulty is not a prerequisite. The full original C0–C9 taxonomy remains useful for subsequent extensions.

The curriculum labels describe geometric difficulty, while milestones describe execution order:

1. **Foundation:** reuse the numerical core, generalize scenes, and qualify the six to eight development cases.
2. **First demonstration:** train on C0–C2 and ordinary C3/C5 cases, then evaluate bounded C8/C9 examples.
3. **Extended capability:** qualify C4/C6/C7, extend C5 toward ten objects, and repeat C8/C9 evaluation with broader feature and arrangement coverage.
4. **Stronger evidence:** selected independent-solver comparisons, repeated seeds, broader held-out shape families, and uncertainty intervals appropriate to independent lineages.

The proof-of-concept demonstration is an early project milestone, not the end of the curriculum. C0–C3/C5 provide the training and collection capabilities needed to test C8/C9 transfer once their declared geometry and material ranges are qualified. The current C3 pair study qualifies only four dielectric circle-pair conditions; it does not qualify mixed materials, general object pairs, or CUDA execution. C4 and C6/C7 remain explicit later stages because they may require changes to the conformal solver or input representation. Do not claim regimes beyond their stage-specific evidence below.

## Stage-by-stage implementation instructions

### C0 — primitive size and position

Use circles, axis-aligned ellipses, and rectangles. Vary size and position across the usable non-PML region; record boundary and observation-contour clearance. Reuse qualified historical training data through the portable reader. Add new examples where the historical coverage is weak.

Verify containment, continuous material sampling, exact-budget projection, and the circle analytic regressions. Produce primitive error-versus-cell curves and a check that teacher meshes offer gains on at least some development conditions. Keep uniform-winning cases. This establishes the numerical and learning baseline for all later stages.

### C1 — orientation, aspect ratio, and curvature

Vary ellipse/rectangle rotation and aspect ratio; use physically consistent transformations of illumination and materials when applying augmentations. Group transformed copies with their parent geometry. Sample the intended orientation range rather than relying on a few near-axis examples.

Check rotated intersections, tangencies, and sampling consistency. Evaluate transfer to held-out transformations without allowing sibling examples into validation/test splits. Deliver accuracy and mesh-savings results by orientation and aspect ratio, including diagonal cases where axis-aligned refinement may lose efficiency.

### C2 — corners, concavity, and changing curvature

Generate diverse triangles, convex/concave polygons, stars, and smooth lobes. Vary vertices, notch dimensions, and boundary curvature within the qualified feature range; repeated rotations of one fixed outline do not supply independent shape diversity.

Verify segment/spline intersections, winding and containment, concave notches, and rejected self-intersections. Improve candidate policies using boundary projections and feature information. Deliver results by shape family and minimum feature size, plus teacher-search headroom on development examples. Train cumulatively with C0/C1 replay.

### C3 — two separated objects

Introduce a collection of two continuous objects, initially with ordinary resolvable gaps. Vary separation, relative orientation, size ratio, and location. Begin with moderate dielectric scenes; add mixed materials only after the simpler collection interface is qualified.

Generalize material rasters, proximity maps, conditioning, monitor enclosure, and candidates to both objects. Verify CPU/CUDA agreement and object-order invariance for nonoverlapping scenes. Deliver accuracy and savings versus separation, size ratio, and projected support. This stage is required for the first distributed-scene demonstration.

### C4 — very small gaps

After C3, run controlled development gap sweeps. Record gap width relative to wavelength, object size, the raster-pixel width, and local cell widths. Compare fixed physical samples along each gap line in addition to far-field scattering so the near-field interaction is explicitly resolved. First establish which gaps the continuous geometry, raster, mesh, and conformal solver can represent consistently.

Keep explicit rejection of unsupported split edges. If handling multiple intersections per edge or another numerical treatment is needed, implement and qualify it as a separate solver change before creating labels. Refine references around strong interactions and check sensitivity to time duration and geometric quadrature. Do not replace the physical gap with a closed or shifted boundary.

Deliver a map of supported/incompatible gap-grid combinations, accuracy versus gap, and the cell/time-step cost of resolving the interaction. If the gap is below the input-raster resolution, version a representation that preserves it before training on that regime. Extend training only over the qualified regime; keep development gap sweeps out of final test claims. Very small gaps are an extension, not a prerequisite for the first C9 demonstration.

### C5 — three to ten mixed objects

Extend the C3 collection interface, first to 3–5 objects for the compact campaign, then to 6–10. Include compact clusters, aligned groups, and dispersed layouts. Use matched-area and matched-object-count controls to distinguish arrangement effects from increased material content.

Measure union area, union projections on both axes, overall extent, minimum separation, and object count. Qualify one monitor enclosing the whole scene with the required vacuum/PML clearance. Deliver savings versus count and arrangement, as well as failure coverage and cumulative-training retention of simpler scenes.

### C6 — holes, cavities, and thin sections

Introduce explicit boundary rings, voids, open cavities, and connected-component information. Define winding, material precedence, and intersection semantics before extending the solver interface; a single filled polygon cannot represent every topology in this stage.

Test holes, necks, cavity openings, and thin material sections against continuous containment and sampling. Qualify any multiple-cut conformal treatment independently. Demonstrate that the raster and projected grid preserve the features used by the solver. Deliver results grouped by topology and feature thickness, with unresolved features identified rather than silently filled or removed.

The first C6 geometry interface is `polygon_with_holes` in scene schema 4, with one `outer_ring_m`, one or more `holes_m`, and a single material on the ring. Ring winding is orientation-independent; holes must be strictly inside the outer ring and may not touch, intersect, or nest. The dielectric sampler, nine-map input, boundary-coordinate candidate maps, area metrics, and minimum feature-size metric consume the continuous rings. Tests cover square rings with wall thicknesses down to 15 mm (about six input pixels), candidate refinement near inner boundaries, and a U-shaped open cavity through continuous membership and rasterization. PEC rings use the existing conformal edge treatment and explicitly reject edges that contain unresolved multiple cuts.

A bounded CPU qualification compared three square dielectric rings with wall thicknesses 90, 30, and 15 mm, plus a U-shaped open cavity. Conditions were one 1 GHz illumination, `εr=4`, conductivity 0.002 S/m, with 0.8–1.2 GHz scattering observables. The rings qualified at 384, 256, and 512 cells/axis, with maximum successive-grid uncertainties of 0.399%, 0.492%, and 0.476%; all independent probes passed. Their 27 uniform/interface/hybrid comparisons at 48/64/96 cells settled and passed axis/grading checks. Results vary by ring and budget: one 30 mm-ring hybrid mesh reached both 2% errors at 96 cells, while uniform was best for the 15 mm ring at 48 and 96 cells. No general nonuniform advantage is established. The open cavity still exceeded the 0.5% complex-field reference gate at 512 cells (0.771% field and 0.578% width change from 384) and produced no mesh labels. Thus C6 has qualified ring evidence, not completed cavity or PEC coverage. Raw data and provenance are in `runs_v2/c6_topology_qualification_v1/c6_topology_qualification_report.json`.

### C7 — multiscale composite geometry

Combine a large body with progressively smaller appendages, notches, gaps, or inclusions. Sweep size ratios over a declared range and record the smallest feature seen by geometry, raster, and grid. Keep the modest domain/electrical-size scope until evidence motivates expansion.

Check whether the 512×512 input resolves the details. If it does not, qualify a revised encoding or higher raster resolution and version the model input; increasing CNN width alone cannot restore missing geometry. Re-profile memory and revalidate references for newly introduced scales. Deliver cell savings, accuracy, and time-step cost versus scale ratio, including representation limits.

### C8 — unseen engineering-like silhouettes

Create or import simplified aircraft-like, ship-like, and vehicle-like outlines with source/template provenance. Use the same continuous polygon/spline machinery as the solver. For the first demonstration, retain ordinary resolvable features and the qualified material range. Add holes, thinner details, and broader electrical sizes only after C6/C7 qualification.

Freeze test templates and group all their transformations before training. Development silhouettes use separate templates. Distinguish unseen templates from entirely unseen construction families in the claim. Freeze the model and geometry-based baseline, then run uniform/heuristic/CNN FDTD at identical conditions.

Deliver individual silhouettes, predicted meshes, scattering curves, error-versus-cell results, and cell-saving tables. Report dielectric and PEC evidence separately. Reusing an inspected test suite for development consumes its held-out status; use a new versioned test cohort for a subsequent independent claim.

### C9 — unseen distributed scenes

Arrange held-out silhouettes and irregular objects at new positions, orientations, and relative scales. Begin with 2–6 objects and ordinary gaps; later add up to ten objects and the gap/topology/scale regimes qualified in C4–C7. Keep material, boundary clearance, and electrical-size ranges explicit.

Include matched-area compact, aligned, and widely dispersed scenes. Report union projections and total cell count so the limitations of tensor-product refinement remain visible. Shared templates across layouts stay in the same split and form a grouped unit in statistical summaries.

Deliver the main sparse-scene evidence: equal-budget accuracy, minimum tested cells for target accuracy, performance versus spatial arrangement, and all incompatible/unresolved cases. If savings disappear when axis coverage grows, report that limit and assess a future local-refinement representation as a separate project change.

## Stage exit gates and evidence

Use these gates to decide whether a stage is ready to feed the next one. They are evidence requirements, not promises of positive results. A stage may exit as **qualified with limits**, **inconclusive**, or **blocked by a documented numerical limitation**; do not turn failed or unsupported cases into training labels.

| Stage | Entry condition | Exit evidence required before expanding |
|---|---|---|
| C0 | Circle solver regressions and portable historical-data reader pass | Valid exact-budget meshes; circle reference agreement; primitive accuracy and cell-saving curves; lineage-safe split |
| C1 | C0 references and data path are stable | Rotated/aspect-ratio coverage; intersection and tangency checks; grouped transformation holdout; performance by orientation |
| C2 | C1 sampling and polygon geometry checks pass | Diverse corners, concavity, and curvature; self-intersection rejection; results by family and minimum feature size |
| C3 | Scene collections, aggregate materials, and monitor enclosure work | Two-object order invariance and CPU/CUDA agreement; controlled separation and size-ratio results; explicit unsupported cases |
| C4 | C3 interaction cases are qualified | Gap sweep states the supported gap-to-cell range; fields and mesh represent gaps consistently; duration/quadrature checks bound near-field error |
| C5 | C3 scene pipeline is stable; C4 restrictions are recorded | 3–10-object coverage across compact/aligned/dispersed layouts; matched controls and scaling results by count and axis projection |
| C6 | Boundary-ring/material-precedence semantics are specified | Qualified hole, cavity, and thin-section geometry and solver handling; feature-preservation checks; unresolved topology cases listed |
| C7 | C2 and relevant C4/C6 features are qualified | Declared multiscale ratio sweep; geometry/raster/grid resolution limits measured; revised encoding versioned if input resolution is inadequate |
| C8 | Training templates are separated from frozen silhouette templates | Frozen unseen-template evaluation with uniform and heuristic controls; per-family results and reference uncertainty |
| C9 | C3/C5 collections are qualified; target templates/layouts are frozen | Frozen sparse-scene evaluation across arrangements, object counts, and axis coverage; savings and limitations reported by lineage |

After each exit gate, freeze the stage manifest, geometry range, numerical protocol, scoring fingerprint, and held-out split before bulk labels are generated. Later-stage results do not retroactively qualify earlier stages. When an implementation change affects geometry semantics, solver behavior, raster channels, or mesh projection, version the affected interface and revalidate the stages that depend on it.

## Rules shared by later stages

- Before expanding a stage, qualify a small development set, estimate its cost, and freeze the geometry range, split, baseline, numerical tolerances, and finite execution budget. Stage sizes are driven by coverage and measured cost; each stage does not require a new large factorial campaign.
- Store manifests, geometry lineage, numerical/scoring/model fingerprints, raw fields, axes, compatibility status, and reference sensitivity. Reuse earlier results only when their fingerprints and physical conditions match. Keep new outputs under `runs_v2/` with stage and protocol versions.
- Continue cumulative training with replay from earlier qualified stages. Fine-tune a checkpoint only when architecture and input semantics are compatible; otherwise initialize or migrate explicitly and verify the migration. Recheck earlier-stage validation to detect lost capability.
- Choose checkpoints and stage changes on development/validation data. Freeze before stage tests. Preserve failed and inconclusive outcomes, and never move difficult test cases into training while retaining the original held-out claim.
- Keep the accuracy objective and weak time-step regularization throughout C0–C9. Numerical/geometry qualification, teacher benefit, learned performance, and campaign completion are distinct results; completion of one does not imply the others.

## Compact experiment

These are initial sizing targets to confirm on development cases before freezing the manifest. The deterministic generator currently produces the exact 64/16/24 lineage counts below. The manifest remains uncreated until the compact cost gate has measured a feasible estimate.

| Component | Starting scope |
|---|---|
| Historical training | Reuse the 1,104 qualified training examples; start with 25% of training draws |
| New training | Approximately 64 independent geometry lineages spanning the foundation and bridge |
| Validation | Approximately 16 separate lineages for profile and FDTD model selection |
| Frozen target test | 12 individual silhouettes and 12 distributed scenes |
| Teacher conditions | One incidence angle and two sampled budgets per new training lineage |
| Teacher candidates | Up to six: uniform, three geometry policies, two seeded perturbations |
| Final conditions | Four square budgets, 32/48/64/96 cells per axis, and two incidence angles |

Distribute sampled training budgets and angles across the corpus so the conditioning range is represented. Use procedural boundary diversity and multi-object layouts, rather than many transforms of a few fixed outlines. All related transformations, object templates, and layout derivatives stay within one split. A template reused across several test scenes remains one grouped source of evidence; 24 scenes do not automatically mean 24 independent shape families.

Final silhouette templates are withheld from training, calibration, and model selection. Validation uses separate procedural geometries. Claim template-level or family-level generalization only when the actual split supports it. Historical validation and previously inspected tests remain development evidence.

At the starting size, new training needs at most `64 × 1 × 2 × 6 = 768` candidate simulations, excluding references, calibration, validation, and testing. This is a work count, not a runtime promise. The former 128-lineage design allowed 12,288 candidate simulations across its full corpus before references.

## Numerical scope and references

- Retain the public `scattermesh` simulation interfaces, 2D TMz physics, continuous geometry, and nonuniform Yee grids. Start with the existing 1.2 m square and 0.8/1.0/1.2 GHz observations. Record sizes in metres and wavelengths.
- Use moderate dielectric contrasts, initially relative permittivities 2 and 4, for the core demonstration. Qualify a small, separately reported PEC supplement. A mandatory 25% PEC quota and the broader material sweep are removed from the first campaign. Freeze any PEC subset before final evaluation and report every rejection.
- Control feature sizes and ordinary gaps on development geometries. Reject unsupported split-edge PEC configurations explicitly. Never move boundaries or close gaps to make a failed mesh pass.
- Use analytic circle references after solver calibration. For other shapes, try cheaper successive uniform resolutions, initially 128/192 and then 256 if needed. These are starting levels, not guaranteed references. Escalate selected difficult cases when the evidence requires it.
- Check spatial and duration sensitivity, and calibrate PML, contour, and material quadrature on a small representative development set. Repeat detailed probes when a case falls outside the calibrated regime or an apparent gain is sensitive to those settings.
- Freeze numerical tolerances, duration limits, and reference escalation rules from calibration before bulk execution. The current field-tail threshold `1e-5` is retained initially. A small tail alone does not establish spectral or spatial convergence.
- Retain teacher alternatives when reference variation makes their ordering ambiguous. Exclude unsettled, incompatible, incomplete, or numerically unsupported labels. Final accuracy and cell-saving conclusions must survive reference variation; otherwise report them as unresolved.

Detailed acceptance and reporting requirements are in [the pilot protocol](docs/v2_pilot.md). A mandatory four-probe 384/512-cell qualification for every condition is superseded by this calibrated, adaptive protocol. Cheaper references still require evidence of adequacy.

## Mesh representation and baselines

The v3 scene interface supports up to ten continuous objects with per-object materials and stable IDs while retaining the single-object adapter. Rasterization, candidate generation, monitor placement, geometry descriptors, and simulation setup use the same object collection. The initial collection path rejects intersecting or touching axis-aligned bounding boxes; C4/C6 need their own numerical and material-precedence work before that restriction can be lifted.

The current nine-map input provides scene-wide material maps, signed-boundary information, pairwise proximity, and coordinates. Conditioning includes object count, area-aggregated material values, feature size, and projection support. Polygon vertices and exact conic extrema feed the interface candidate policy. Recheck memory and freeze feature definitions before training.

The CNN predicts x/y densities for a tensor-product grid. Refining an axis affects whole grid lines; sparse occupied area alone does not guarantee savings. Measure occupied area, the union of object projections onto each axis, overall scene extent, and object count separately. Compare compact, aligned, and widely distributed layouts at similar occupied area. The old 35% support restriction is not a universal constraint on distributed scenes.

Evaluate three methods under identical physical conditions and cell budgets:

1. Uniform mesh.
2. A fixed geometry-based mesh policy, with its rule and parameters selected on development/validation data and frozen before testing.
3. The frozen CNN mesh.

Use the best candidate found by teacher search on development cases to measure the available improvement and how much the CNN recovers. Candidate policies must respond to actual boundaries and all objects, not only the bounding box of one object. Final test cases do not require a full teacher search. Keep uniform-winning cases, invalid meshes, and inconclusive comparisons in the report.

## Learning objective and model

Retain the current scoring rule:

```text
L_accuracy = L_complex + 0.25 L_width
J = L_accuracy * (dt_uniform / dt_candidate)^0.05
```

The uniform baseline uses the same scene and cell budget. Teacher selection and set-valued profile distillation use valid physics-scored candidates; physical accuracy determines checkpoint selection and the main conclusions. Re-score saved candidates at exponents 0, 0.02, 0.05, and 0.1 without repeating FDTD. Preserve numerical, scoring, and model fingerprints separately.

Retain the large five-level residual U-Net: widths 64/128/256/512/512, GroupNorm, SiLU, skip connections, bottleneck conditioning, and eight-head attention. The current implementation has **26,850,497 parameters**. Its nine 512×512 input maps are an image representation, independent of the FDTD grid resolution. Keep exact requested cell counts, positive widths, grading at most 3, and recorded projection repairs.

Initialize afresh and reuse data. Retain AdamW at `3e-4`, weight decay `1e-4`, FP16, and effective batch size 32 with accumulation. The existing training-step probe admitted microbatch 8 at 16.16 GiB; recheck if the scene encoding changes its memory use. Report the final parameter count if conditioning dimensions change.

Replace the rigid 120-epoch C0–C2 schedule with a short foundation warm-up followed by cumulative training including separated objects. Start from 10 warm-up epochs and up to 50 cumulative epochs, with validation patience 10 in the cumulative phase; freeze the schedule after development profiling. Evaluate the three best validation-profile checkpoints with validation FDTD and freeze one before target testing.

The existing width-16 model has 1,683,761 parameters. Its comparison is a follow-up ablation after the main demonstration, subject to the remaining resource budget.

## Implementation sequence and execution limits

**Backend rule:** every new FDTD profiling, reference, teacher, validation, and test simulation runs through the Cython-bound, `nvcc`-compiled cooperative CUDA kernel. The entire time loop, including CPML, conformal PEC handling, source updates, field checks, and DFT accumulation, stays on device without host/device transfers. CPU work is limited to geometry/coefficient preparation before launch and collecting/processing results afterward. Legacy NumPy and Torch solvers may be used only for parity regressions and the archived speed benchmark; they cannot generate campaign labels or evaluation evidence. A missing or stale compiled extension is a hard error, never a silent fallback. Rebuild with `.venv/bin/python setup_cuda.py build_ext --inplace` when CUDA/Cython sources change.

1. Generalize scenes, rasterization, conditioning, candidates, and simulation entry points to multiple objects. Add simplified silhouette definitions with explicit lineage. Preserve circle regressions and verify representative new CPU/CUDA cases.
2. Make counts, split membership, candidate sets, per-split conditions, reference rules, and phase deadlines configurable. Add the fixed heuristic baseline and reference-sensitivity reporting.
3. Run the six to eight development cases. Measure mesh benefit, numerical compatibility, reference sensitivity, and full per-condition cost. Freeze the protocol and final test manifest before collecting test results.
4. Generate the compact teacher set, train the large CNN, select with validation FDTD, and evaluate the frozen silhouette and distributed-scene suite. Save raw fields, axes, statuses, uncertainty estimates, and provenance.
5. Produce the required report. Add the small-model ablation and broader material cases after the main result if resources allow.

Retain a **24-hour upper limit** for the first revised automated campaign after implementation and calibration. Use the available four GPUs for independent FDTD workers. Allocate phase budgets from the new measurements, reserving time for final evaluation and reporting; the old mandatory 14-hour data gate is superseded. Calibration and implementation time are tracked separately.

Before bulk launch, freeze a feasible workload. Reduce new training/search work if necessary while preserving both target cohorts; record any revised counts before model training and test evaluation. Do not start an oversized workload on the assumption that partial completion will suffice. Save resumable state at deadlines and report unfinished cases explicitly. No runtime estimate for this revised campaign exists yet.

## Required deliverables and interpretation

- Error versus cell count for uniform, fixed heuristic, and CNN, including complex-field and scattering-width errors separately.
- Minimum tested cell count meeting both 5% and 10% error targets; retain 2% as an optional tighter target when references support it. Report unmet and unresolved targets without interpolation beyond tested budgets. Cell-saving ratios use total cells `Nx × Ny`.
- Results for individual silhouettes and distributed scenes separately, with geometry/material groups, valid coverage, medians, and individual outcomes. Count related layouts and angles within their parent lineage.
- Meshes and scattering curves showing representative gains, uniform-winning cases, invalid predictions, and unresolved comparisons. Label any post-inference repair or fallback separately from direct CNN output.
- Cell savings versus axis-projection coverage and spatial arrangement; reference sensitivity; time step, update count, runtime, memory, and inference cost. Distinguish one-time training/teacher-search cost from per-scene mesh generation.
- Counts of incompatible, unsettled, reference-unresolved, budget-limited, and incomplete conditions. Separately report PEC results and the optional model-size comparison.

A positive proof of concept requires repeatable, reference-resolved improvements on held-out examples in **both** silhouette and distributed-scene cohorts. Summarize the full frozen suite; a few selected mesh pictures are insufficient. Comparison with the fixed heuristic establishes whether the CNN adds accuracy or can recover useful adaptation cheaply. A CNN that only matches the heuristic supports a narrower automation claim.

The former C0–C2 rule of 75% of dielectric conditions improving by at least 5% with a median ratio of 1.2 is no longer the project-wide success gate. Report effect sizes, coverage, and uncertainty directly. Limited or negative outcomes lead to a scoped remediation report without retuning on the frozen test set. Broader claims require further geometries, repeated seeds, larger independent test sets, and selected independent-solver comparisons.
