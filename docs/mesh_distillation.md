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

## Set-valued targets

`scripts/build_mesh_distillation_dataset.py` refuses partial campaigns. The source
report must identify the exact campaign, have decision `accepted`, and show every
candidate case accepted. The builder checks each saved axis against its requested
Nx and Ny and independently recomputes the soft-Nt ranking.

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
