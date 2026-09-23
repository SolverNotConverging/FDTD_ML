# Historical mesh distillation

This document describes the first distillation dataset. The active v2 teachers,
weak time-step score, and portable legacy import are in [v2 pilot](v2_pilot.md).

# Exact-budget mesh distillation

The first learned mesher is a 532,593-parameter residual U-Net. It predicts one
positive 2D importance map, then obtains realizable x and y density profiles by
log-sum-exp projection along the opposite axes. It never predicts an unconstrained
2D mesh mask.

## Input contract

The 128x128 raster has five channels:

1. geometry occupancy;
2. logarithmically normalized relative permittivity inside material;
3. logarithmically normalized electric conductivity inside material;
4. clipped signed distance, scaled by feature size;
5. an interface-proximity channel at the raster-pixel scale.

Global conditioning supplies incidence sine/cosine, requested Nx and Ny, frequency
band limits, feature size, maximum material permittivity and conductivity, and the
selected grading limit. Feature size and budget are explicit inputs rather than
quantities the CNN must infer from a low-resolution mask.

## Sparse-scene input and capacity study

The sparse-scene `sparse_v2` input has seven spatial maps: dielectric fill,
normalized log permittivity, normalized electric conductivity, PEC fill, signed
distance to the nearest object boundary, boundary proximity, and pair proximity.
The last map responds to nearby object pairs; it is **not yet a full local
clearance/feature-size field**. The 15 global conditioning values include incidence,
exact x/y budget, frequency band, feature size, maximum material contrast,
grading limit, object count, PEC fraction, occupied area, and projected support.

The channel count of an internal convolution is model capacity, not a requirement
for additional physical input fields. For example, the `r256_c32` variant reads
seven 256x256 physical maps and its first learned convolution produces 32 feature
maps. The nine-variant study crosses raster sizes 128/256/384 with base widths
16/24/32. Every variant receives the same seven maps and 15 conditioning values.

The initial joint dataset combines the single-circle campaign with qualified
dielectric-pair and mixed dielectric/PEC pilots. Sparse scenes receive 70% of
training samples. Its sparse pool is still small, so this study compares capacity
and input resolution; it does not establish sparse-scene generalization. The
follow-up 96-geometry pair campaign contains separate train, validation, and test
scenes across dielectric circles, mixed dielectric shapes, dielectric/PEC pairs,
and PEC rectangle pairs. It first requires all 192 fine references to settle and
pass a 192/256 complex-field convergence check, then searches 2,016 exact-budget
candidate meshes. All nine trained variants are evaluated by held-out FDTD
scattering before a capacity choice is made.

## Set-valued targets

`scripts/build_mesh_distillation_dataset.py` refuses partial campaigns. The source
report must identify the exact campaign, have decision `accepted`, and contain one
terminal record for every candidate. An accepted uniform baseline is mandatory for
each condition. A nonuniform candidate that remains unsettled at the hard duration
limit is retained for provenance, but a saved validity mask excludes its profile
and score from the set-valued loss. The builder also requires every pre-M5
label-diversity check to pass, verifies each saved axis against its requested Nx and
Ny, and independently recomputes the soft-Nt ranking.

Each candidate axis becomes probability mass on 128 fixed spatial bins. Every mesh
cell contributes equal mass, distributed uniformly across its interval. The
artifact retains all six candidate profiles and their physics scores instead of
collapsing them to one arbitrary axis. The training loss combines CDF error to the
selected best candidate with a physics-score-weighted loss over the complete set.

## Exact projection

Inference treats each predicted profile as a piecewise-uniform probability density
and places exactly Nx+1 and Ny+1 nodes at its quantiles. If adjacent cell-size ratios
exceed the configured limit, the profile is mixed toward uniform by deterministic
bisection until the projected axis passes. The repair fraction is stored for every
axis. A zero fraction means the raw CNN output was already feasible.

Distillation metrics do not establish physics accuracy. After training,
`scripts/run_learned_mesh_pilot.py` reruns the CUDA solver on validation and test
meshes, applies the same 70/140/560 ns settling schedule, and compares the learned
soft-Nt score with both uniform and the best searched teacher candidate. Only that
physics evaluation can promote the model beyond the M5 pilot.

The first profile-learning result is recorded in the
[mesh-distillation pilot report](validation/mesh_distillation_pilot.md), and its
held-out CUDA result is in the
[learned-mesh physics pilot](validation/learned_mesh_physics_pilot.md).

## Full-stage handoff

`scripts/continue_exact_mesh_pipeline.py` waits until every candidate record is
terminal, including all declared duration retries, reruns the frozen label gate,
builds the immutable set-valued dataset, and trains the full checkpoint. It records
hashes of the dataset, configuration, training entry point, and every active
`scattermesh` source module.

`scripts/continue_learned_mesh_evaluation.py` then waits for that checkpoint and
runs all validation and test examples through the CUDA solver. Cases are assigned
to GPUs by a stable hash of the case ID, so dataset ordering cannot concentrate one
budget or material tier on a worker. Failed worker processes have a fixed retry
limit and reuse fingerprint-validated cached attempts. The final report compares
the learned mesh with both exact-budget uniform and the best searched teacher and
records split-level tail statistics before declaring the frozen physics gate. Its
`evaluation_summary.png` plots learned versus uniform scores, improvement and
teacher gap by exact budget, and complex-field versus log-RCS improvement.
