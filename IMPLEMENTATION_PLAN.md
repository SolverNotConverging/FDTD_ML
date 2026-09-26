# Mesh-CNN v2: C0–C9 curriculum for nonuniform resolution demand

Revised 23 September 2026. This is the authoritative project-wide plan. **C0 acquisition is complete and fresh large-CNN training is active.** The previous 64/16/24 compact campaign and its 2.92-hour estimate are superseded as workload commitments. Completed results remain development evidence.

The sole project root is `/home/s2307298/projects/FDTD_ML`; new outputs belong under `runs_v2/`. Original scattering and receiver projects remain in `archive/`. Previous planning documents are preserved in `archive/plan_revisions/2026-09-23_before_sparse_curriculum_restart/`.

## 1. Objective and scope

Demonstrate that a CNN can rapidly allocate a fixed mesh budget where **electromagnetic resolution demand varies across a 2D scene**, learning from numerically optimized teachers. This includes isolated/distributed objects in free space and highly occupied heterogeneous material regions. The final scope includes bird-like, warship-like, aircraft-like, satellite-like, and radio-telescope-like silhouettes, several material regions within a body, and mixtures of these objects.

The primary outcomes are better complex-field and scattering-width accuracy at equal spatial cell count, and fewer cells meeting the same accuracy threshold. Time step, update count, runtime, memory, inference cost, and one-time teacher-search cost accompany those outcomes. The time-step penalty stays weak.

Restart the large CNN from a fresh C0 initialization and continue cumulatively. Simple shapes are substantive manuscript experiments: position, size, rotation, concavity, and material sweeps must remain visible in the final results. Reuse historical data only when its physics, provenance, qualification, and split membership match the new protocol.

Use idealized 2D TMz geometries at declared electrical sizes, initially in the existing 1.2 m square at 0.8/1.0/1.2 GHz. Object names describe silhouette families; they do not establish validated biological tissue models, full-size platforms, CAD accuracy, or 3D scattering. Broader frequencies, electrical sizes, and realistic platform scale ratios require separate qualification.

## 2. Resolution-demand concentration, entropy, and spatial layout

The organizing principle is **nonuniform electromagnetic resolution demand**, not a maximum material-coverage percentage. A region fully occupied by low-permittivity material with a small high-permittivity inclusion can offer refinement opportunity. A uniformly demanding region can offer little, even though its material is complex. Keep both as controls. Low-area objects in free space remain an important subclass, initially sampled around 0.5–8% occupied area, rather than a universal admission rule.

Develop a positive spatial demand proxy `d(x,y)` on a fixed physical region and analysis raster. Candidate ingredients are local wavelength at the maximum modeled frequency, dielectric/PEC and internal interfaces, narrow gaps, curvature, and smallest features. For nonmagnetic lossless dielectrics, wavelength scales as `1/sqrt(epsilon_r)` ([resolution background](https://meep.readthedocs.io/en/latest/FAQ/)). Treat PEC demand through boundary geometry, not an infinite dielectric value. Loss is not automatically a refinement bonus; its influence on propagation, attenuation, and settling requires calibration. Include a propagation/PML floor so nominally unimportant regions are not discarded.

Use an entropy-based diagnostic of this demand map:

```text
p_i = d_i / sum_j(d_j)
H_d = -sum_i(p_i * log(p_i))
effective_demand_fraction = exp(H_d) / N
```

Here `N` is the number of equal-area analysis bins. A nearly uniform demand distribution has effective fraction near 1; concentrated demand has a smaller fraction. This measures concentration, not absolute difficulty: also report demand magnitude, contrast, and feature scales. Freeze normalization, baseline floor, smoothing scale, and analysis region before comparisons. Validate raster-resolution sensitivity. This is a proposed descriptor, not a calibrated error estimator or a guarantee of mesh savings.

Also report concentration of **excess demand above the smooth-background requirement**, using `e_i=max(d_i-d_base,i,0)` and normalizing only when its sum is positive. This prevents the ubiquitous baseline propagation requirement from concealing a small region that needs much finer cells. Record the total excess and its ratio to baseline demand alongside its entropy. If there is no excess, report that explicitly; an all-zero map must not be assigned a misleading concentration score. Neither total nor excess entropy replaces the physical-error comparison.

Do not use material-histogram entropy alone. A block and a finely alternating arrangement can have identical material fractions but different interface scales and mesh requirements. Demand entropy itself is also insensitive to rearranging identical demand values. Therefore retain interface density, spatial arrangement, and axis-specific demand measures. For tensor-product grids, assess conservative line envelopes such as `d_x(x)=max_y d(x,y)` and `d_y(y)=max_x d(x,y)`, their effective fractions, and integrated marginals. A low 2D effective fraction with broad x/y demand may still yield limited tensor-product savings.

Report occupied union area, region/object counts, x/y geometric projections, enclosing extent, minimum gap/feature, and electrical sizes as complementary descriptors. Define the physical region of interest separately from PML/buffer space and report total solver cells for savings. Do not manufacture favorable percentages by changing the reporting region or padding across mesh methods.

Acquire three controlled scene types as their numerical representation becomes available: low-area objects in free space; highly occupied low-demand material with localized high-demand regions; and broadly distributed/high-demand controls. Match material fractions while changing block/layer/dispersed layouts to test whether the descriptors predict the *measured* optimized-teacher advantage. Correlate proposed entropy measures with teacher gains on development data before adopting thresholds. The physical FDTD error remains the teacher objective; the CNN is not trained merely to lower entropy.

True non-vacuum background filling the domain requires an additional numerical qualification: the current analytic incident source and far-field transform assume vacuum. Generalize background wave speed/impedance, source contrast, PML, observation normalization, and reference solutions consistently before claiming that regime. Dense material inside a vacuum observation buffer and a domain-filling material background are different numerical cases. A near-to-far contour must remain within the qualified homogeneous exterior ([near-to-far background](https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/)).

Keep domain, continuous geometry, material, source, observation protocol, and budget identical between methods. Retain uniformly demanding and unfavorable-layout cases to measure where nonuniform meshes cease to help.

## 3. Revised stages and proposed dataset sizes

Counts are **new geometry lineages across all splits**, before transformations, material variants, angles, and mesh budgets. They are planning targets, not a runtime promise or an immediate launch. Initially split complete lineages 75%/12.5%/12.5% into train/validation/test. All descendants and related material/layout variants stay with their parent lineage.

| Stage | Geometry and difficulty | Proposed new lineages | Required demonstration |
|---|---|---:|---|
| C0 | Isolated circle, ellipse, rectangle; size and position; homogeneous materials | 512 | Analytic circle checks, initial learning curves, translation/scale/material response |
| C1 | Rotated and elongated primitives; aspect ratio and curvature | 512 | Orientation/aspect-ratio generalization and retained C0 performance |
| C2 | Triangle, convex/concave polygon, star, smooth lobes | 1,024 | Corners, notches, changing curvature, error by minimum feature size |
| C3 | Two separated objects with independent size, position, orientation, material | 512 | Same-material and mixed dielectric/PEC pairs; allocation between objects |
| C4 | Controlled close pairs and gap sweeps in an otherwise sparse domain | 256 | Near/far fields versus gap; qualified resolution and unsupported cases |
| C5 | 3–10 mixed primitive/irregular objects; compact/aligned/dispersed layouts | 768 | Savings versus object count, projected coverage, and relative scale |
| C6 | Holes, cavities, thin sections; simple two-region bodies and internal interfaces | 512 | Topology preservation and material-partition qualification |
| C7 | Multiscale assemblies with several material regions and small appendages | 768 | Refinement of internal interfaces, small high-contrast regions, and fine features |
| C8 | Bird, warship, aircraft, satellite, radio-telescope silhouettes; homogeneous then composite | 500 initially, balanced by family | Transfer from simpler geometry, then unseen-template performance after silhouette training |
| C9 | Sparse mixtures of target silhouettes and irregular objects, initially 2–6 then up to 10 | 768 | Unseen combinations, materials, layouts, and relative scales |

Acquire balanced batches of 128–256 lineages. First calibrate 32–64 C0 physical conditions. Increase or stop acquisition using validation learning curves, teacher-search plateaus, numerical coverage, and measured cost; do not generate the whole table in one job.

For training, start with 4–8 sampled material/angle/budget conditions per lineage, balanced over the corpus. Reserve controlled development panels for full matched sweeps. Avoid a full Cartesian product, but **optimize a teacher for every sampled physical condition**. Start with exact square budgets of 32/48/64/96 cells per axis; add 128 where justified and implemented. The CNN raster resolution is independent of the FDTD grid.

### C0–C2: establish and document the mechanism

Cover lossless dielectric permittivities 2, 4, 8, and 12 and PEC. Prescribed lossy materials are separate physical cases; additional loss used solely for settling follows Section 5. Match selected geometries across materials so the teacher response is visible. Retain low-contrast cases where uniform wins.

Preserve exact conic/polygon intersections and smooth continuous boundaries. Verify rotation, tangency, containment, concavity, sampling, signed distances, and exact budgets. Teacher search leaves physical boundaries fixed. Qualify representative compiled-CUDA results against analytic circles and existing parity regressions.

Deliver a shape/position/scale/material atlas, optimization curves versus FDTD calls, uniform/teacher/CNN mesh examples and scattering curves, error-versus-budget plots, and stage learning curves. Keep a locked validation panel for forgetting checks after each stage. Open stage tests only after freezing that stage's model and protocol; subsequent tuning requires a new held-out cohort.

### C3–C5: allocate the mesh between objects

C3 introduces independently described objects, including dielectric/dielectric and dielectric/PEC pairs. Materials are spatial inputs, not inferred from shape category. Include cases where a small high-contrast object deserves disproportionate refinement.

C4 varies ordinary gaps down to qualified small gaps. Record gap/wavelength, gap/raster-spacing, and gap/mesh-spacing. Use fixed physical near-field probes as well as far-field observables. Reject unsupported multiple PEC cuts on one edge; never close gaps or move boundaries to make a grid pass.

C5 increases count and material diversity while measuring concentration of resolution demand. Use matched-area, matched-material, and matched-object controls across arrangements. Begin with moderate size ratios and broaden them only when inputs and references resolve the small objects. Report local development-probe errors alongside global scattering so dominant scatterers do not hide a poorly resolved small component.

### C6–C7: qualify material regions before complex silhouettes

The current separated-object interface is **not yet a general representation of touching, nested, or partitioned material regions**. Implement a versioned body/region schema: body and region IDs, geometry lineage, continuous boundaries, materials, shared interfaces, holes, and provenance.

Require a unique material assignment at every point. Regions may share an exact interface; overlapping interiors and material precedence require explicit semantics and tests. An inclusion replaces the corresponding host volume instead of double-counting two filled objects. Mixed PEC/dielectric interfaces need separate conformal qualification. Preserve unsupported configurations as failures.

Generalize membership/intersections, coefficient preparation, material sampling, bounds, area accounting, and feature descriptors around the same representation. Start with layered rectangles, concentric partitions, cavities, and simple metal/dielectric interfaces. Include highly occupied low-permittivity hosts with localized high-permittivity inclusions, and matched material fractions arranged as blocks or fine layers. Existing ring studies are limited evidence, not completion of these stages.

Version the CNN inputs to expose local permittivity, conductivity/loss, PEC occupancy, outer and internal interfaces, narrow features, and coordinates. Global average material values alone cannot describe a composite body. The current nine-map encoding is a starting point; recheck raster feature preservation and GPU memory when it changes. Features below the representation's resolution require a declared representation change.

### C8: five silhouette families

Create/import continuous outlines with template provenance for bird bodies/wings, ship hull/superstructure outlines, aircraft bodies/wings/tails, satellite bodies/panels/antenna-like appendages, and telescope dish/support outlines. Restrict topology and feature thicknesses to the qualified C6/C7 range.

Start with homogeneous idealizations, then add a few interpretable regions: a prescribed lossy dielectric bird surrogate; a PEC aircraft with dielectric regions; a PEC hull with a dielectric superstructure region; metal/dielectric satellite parts; and a PEC reflector with simplified supports. These are declared approximations. Real biological or frequency-dependent composite claims require sourced material data and a separately qualified dispersive model.

Separate two experiments: first freeze the C0–C7 model and evaluate transfer to a reserved silhouette cohort; then train on separate C8 templates and evaluate unseen instances within each family. If the first evaluation informs adaptation, use a separately reserved cohort for the adapted model. Entire-family holdouts are an optional stronger test. Do not call a trained family wholly unseen.

Use controlled material reassignment on identical shapes to establish that the CNN responds to electromagnetic properties rather than category identity.

### C9: mixed sparse scenes

Compose examples such as lossy bird surrogates with PEC aircraft, aircraft with satellites, ships with birds, or telescope-like structures with other scatterers. These are idealized 2D scattering compositions; real-world distances and platform size ratios are not required for the first proof of concept.

Begin with 2–6 objects, then up to 10. Vary position, relative scale, angle, material assignment, and spacing across declared demand-concentration and occupancy strata while respecting numerical clearances. Group shared template/layout ancestry across splits. Test compositional transfer before C9 adaptation, and use separate training scenes if adaptation is needed.

Final results must include held-out templates/combinations, material assignments, and matched compact/aligned/dispersed controls. Keep broad-projection sparse scenes visible as limits of tensor-product meshing.

## 4. Teacher optimization and distillation

A condition comprises continuous geometry, all material regions, illumination/frequency configuration, exact x/y cell counts, and numerical protocol. Each new train/validation condition receives an independent optimization, including PEC and low-contrast cases. Final test conditions cannot guide teacher design, training, or model selection.

An optimized-teacher comparison on a final test condition may be computed only after the CNN, search protocol, and evaluation choices are frozen. Report it as an expensive per-case oracle diagnostic; its labels and rankings cannot feed training or model selection for that cohort. The mandatory optimize-then-distill step applies to every sampled learning condition.

Use the implemented seeded differential-evolution search as the initial method. It is a bounded derivative-free population search whose evaluation cost must be measured ([SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)). Call its result the **best found teacher**, not a guaranteed global optimum.

1. Qualify the reference and accepted uniform baseline under a common observation and settling protocol.
2. Seed from uniform, geometry/interface policies, material-aware proposals, and deterministic perturbations. Implement local-wavelength and internal-interface information in the seeds.
3. Begin with the implemented 12 smooth log-density parameters and 96 evaluations. Compare 96/192/384 evaluations on development data; choose search budgets from improvement-versus-cost curves. Richer local bases and multiple restarts are planned for complex C6–C9 cases.
4. Project every proposal to exact counts, positive widths, and adjacent ratios at most 3. Record repairs, preserve geometry, and reject incompatible or unsettled candidates.
5. Rank with `L_accuracy = L_complex + 0.25 L_width` and `J = L_accuracy * (dt_uniform / dt_candidate)^0.05`. Principal comparisons use raw physical accuracy. Re-score exponents 0/0.02/0.05/0.1 from saved fields without new FDTD.
6. Save evaluated axes, spectra, statuses, dt, cost, and provenance. Retain several distinct good targets and uniform/fixed-policy controls; keep alternatives if reference changes make their ordering ambiguous.
7. Distill the profiles into the CNN and report the gap between CNN and teacher physical accuracy. Uniform is a valid target when search does not resolve an improvement.

Freeze search settings per acquisition batch. Save optimizer population/RNG state or verify deterministic replay from cached objective records, including failed cases. Interrupted searches stay incomplete and cannot promote partial trials as completed labels. Keep numerical fingerprints separate from optimizer/scoring/model fingerprints.

The current implementation covers a 96-trial pilot and condition-specific worker promotion. Broader search-quality calibration, failed-trial resume, and the revised stage scheduler still require implementation or verification.

## 5. Materials, settling, and references

Keep lossless high-permittivity conditions whenever they settle within the allowed maximum physical duration. Long-lived resonance, spatial nonconvergence, unstable updates, and execution-budget exhaustion are distinct failure modes.

Prescribed loss is legitimate input physics, including a bird surrogate. **Additional loss used solely to help settling is conditional**:

1. Run the original lossless condition through the frozen duration ladder, currently 70/140/560 ns, with finite-field checks and the `1e-5` tail gate. Freeze any revised ceiling before acquisition.
2. Retain lossless material if it settles. If one candidate fails while the condition/reference is otherwise valid, reject the candidate rather than changing the material of the whole comparison.
3. If the condition remains unsettled at the allowed ceiling, preserve its failure and optionally create a separately identified lossy variant. Keep its geometry lineage/split and record conductivity, reference frequency, loss tangent, reason, and parent condition.
4. Recompute reference and all compared meshes for that material. It cannot establish lossless performance or be compared against a lossless uniform baseline. Added loss cannot repair unresolved geometry or spatial convergence.

For the SI dielectric implementation, report `tanδ(f) = sigma / (2*pi*f*epsilon0*epsilon_r)`. Fixed conductivity implies frequency-dependent loss tangent. Conductivity is a limited-band material approximation; realistic dispersion is a later numerical extension ([material-model background](https://meep.readthedocs.io/en/latest/Materials/)). Do not import another solver's conductivity unit convention into this SI code.

Use analytic references for qualified circles and adaptive uniform refinement for other shapes: 96/128/192/256 initially, with selective 384/512 escalation. Check duration, quadrature, contour, and PML variation on representative shape/material/region classes. The former compact 1% field/2% width spatial gates are exploratory defaults, not proof that every teacher ranking is resolved.

Keep reference variants and uncertainty beside scores. Refinement must resolve a claimed gain or threshold crossing; otherwise preserve ambiguity, escalate within the cap, or exclude the label. Report geometry incompatibility, unstable/nonfinite fields, unsettled fields, spatial nonconvergence, budget exhaustion, and interruption separately.

## 6. CNN training and split discipline

Retain the large residual U-Net: widths 64/128/256/512/512, GroupNorm, SiLU, skip connections, conditioning, and eight-head bottleneck attention. The present nine-input model has 26,850,497 parameters. New material-region channels require a versioned parameter count and memory check.

Initialize afresh at C0 and continue cumulatively. Replace the old ten-epoch C0 warm-up with a real C0 block: propose up to 80 epochs, then 40–60 additional epochs per new stage, with validation patience 15. Calibrate these caps using learning curves; data and teacher quality take precedence over fixed epoch counts.

Start AdamW at `3e-4`, weight decay `1e-4`, effective batch 32, FP16, and accumulation. Re-probe microbatches 1/2/4/8 below 20 GiB after encoding changes. Use validation-controlled learning-rate reduction during continuation. After C0, start with 50% current-stage and 50% balanced earlier-stage replay. Qualified historical examples may contribute up to 25% of total draws within this mixture.

Group transformed copies, material variants, and template/layout ancestry before splitting. Inputs carry local materials, illumination, frequency/electrical scale, and requested counts. A region-encoding change may require an explicit model migration rather than silently loading incompatible weights.

Select three best profile checkpoints with validation FDTD, freeze one, and evaluate earlier validation panels for forgetting. Open stage/final tests only under a frozen protocol. Use fresh reserved cohorts for new claims after results influence development.

Add the width-16 comparison model (currently 1,683,761 parameters) and teacher-budget ablations once the C0–C2 mechanism is established. Repeat training seeds for manuscript confidence; a single seed remains preliminary evidence.

## 7. Implementation order and execution budgets

**Every new FDTD run uses the Cython-bound, nvcc-compiled cooperative CUDA kernel.** Updates, CPML, conformal PEC, source, DFT, and checks stay on GPU for the entire time loop with zero host/device transfers during stepping. CPU preparation and final processing are allowed. Legacy CPU/Torch solvers are restricted to parity tests and saved comparisons; missing/stale compiled code is a hard error.

The user resumed execution on 23 September 2026. The first C0 calibration and independent controls are complete. A corrected 128-lineage C0 acquisition completed under `runs_v2/c0_restart/acquisition_002`, passing the qualified-label gate with 68 training and 10 validation lineages. Fresh large-CNN training is active, and validation FDTD selection is queued. Preserve completed pilots and optimizer records as development evidence.

Current implementation sequence:

1. Finish large-CNN C0 training and select its checkpoint on qualified validation FDTD. Keep the earlier compact launch commands outside the active C0 workflow.
2. Add C1–C9 manifests, demand/occupancy/layout acquisition strata, material/geometry lineage groups, stage checkpoints, and cumulative replay. The C0 worker and training state are restartable; later-stage scheduling remains planned.
3. Extend the completed 36-condition C0 calibration with 96/192/384-search plateaus, reference sensitivity, material settling, and measured training throughput where the available evidence is thin.
4. Select the C0 model on validation FDTD, freeze it, open the untouched C0 test, and produce manuscript panels. Expand toward 512 C0 lineages only when costs and learning curves support it.
5. Advance through C1–C5; implement/qualify region partitions before C6/C7; then build C8/C9 training and held-out cohorts under the split rules.

Use four GPUs for independent condition workers and separate GPUs for training where useful. Execute checkpointed batches, initially at most 24 hours each, with phase deadlines and saved state. The whole C0–C9 curriculum is a multi-batch effort. The former compact 2.92-hour projection does not apply to it. Report measured forecasts and revised acquisition sizes before bulk execution.

## 8. Manuscript deliverables and stage gates

| Evidence | Required content |
|---|---|
| Mechanism, C0–C2 | Primitive/concave shapes at varied position, scale, orientation and material; teacher/CNN meshes; learning and optimizer curves |
| Accuracy and savings | Raw complex-field/width errors versus cell count; equal-budget uniform/heuristic/teacher/CNN; minimum tested cells meeting both 2%, 5%, 10% targets |
| Materials | Matched geometry across permittivities and PEC; prescribed loss; settled lossless high-εr; conditional lossy variants separately |
| Resolution-demand distribution | Savings versus demand entropy/effective fraction, absolute demand, area, interface density, and x/y demand coverage; matched layouts, dense heterogeneous cases, and uniform-demand controls |
| Composite bodies | Qualified interfaces, cavities and fine features; homogeneous/composite comparisons; material reassignment controls |
| Target transfer | Five silhouette families and mixed scenes; before/after silhouette training as separate experiments; template/composition holdouts |
| Cost and reliability | dt, updates, runtime, memory, inference and teacher cost, reference uncertainty, incompatible/unsettled/unresolved/incomplete counts |
| Robustness | Large/small model, teacher budget, dt exponent, repeated seeds, confidence intervals grouped by geometry/template lineage |

Before stage advancement, require a declared supported numerical range, reference-qualified teachers across material/sparsity strata, and validation evidence that the CNN recovers teacher gains without unacceptable forgetting. Set quantitative criteria from calibration before opening that stage's tests. If teachers fail, investigate physical regime and mesh parameterization; if teachers help but the CNN fails, investigate data, encoding, and distillation. Preserve uniform-winning and failed cases.

Before stronger manuscript claims, add independent solver comparisons, repeated seeds, larger held-out cohorts, and uncertainty-aware statistics. Campaign completion and positive scientific evidence remain distinct outcomes.
