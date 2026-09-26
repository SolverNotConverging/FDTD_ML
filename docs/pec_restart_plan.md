# PEC restart: implemented foundation and next gates

Updated 2026-09-27. The original proposal and complete legacy source are preserved
on `legacy-project` at `17b1fb7f83feb95ea102417a60cd58f8c6f8136e`.
The user superseded the original staircase phase: conformal FDTD and enlarged
cells are mandatory, with GPU-resident DFT, NF2FF, and device-controlled stopping.

## Implemented stages 0–2

| Stage | Implementation | Evidence |
|---|---|---|
| 0: preserve and simplify | Legacy branch; work remains on master; obsolete dielectric/training code removed; mesh projection and CPML retained; native CUDA rebuilt | Mesh regression suite and CUDA build |
| 1: scattering foundation | +x TFSF from a matching 1D nonuniform incident solver; staggered E/H DFTs; GPU NF2FF; multiple bins; asynchronous telemetry; safety-capped device auto-stop | CPU/GPU parity, analytical cylinder, empty-domain leakage, incident phase, contour/PML/time/precision checks |
| 2: boundary fidelity | Exact circle/polygon intersections; conservative disjoint enlarged-cell pairs; spatial-operator timestep bound; unsupported topology rejected | Tiny-cut long run, conservation/SPD tests, cylinder refinement, translation/radius tests, paired and slotted objects |

See [validation](validation.md) for measurements and limitations. These stages
provide a qualified small-case research foundation, not a universal accuracy
guarantee or a training dataset. In particular, corners, narrow gaps and high-Q
objects need geometry-specific resolution and convergence checks.

## Stage 3: determine whether mesh allocation has useful headroom

Compare uniform, tuned geometry-rule, and solver-searched meshes under identical
physics. The existing grid family is `x[i] × y[j]`: refinement adds entire rows
and columns. CNN outputs should remain positive 1D densities `rho_x`, `rho_y`,
projected to exact budgets with anchors, grading, minimum spacing and fixed collars.

Start with 12–20 continuous base shapes, one incidence direction, several
electrical sizes and budgets. Include circles, rotated polygons, wedges, paired
conductors, gaps and slots. Add explicit feature anchors or reject candidates
whose cut topology cannot be represented; never silently change the geometry.
Develop a stronger geometry density than the demonstration Gaussian density.

Build reference gates before search: refine space, inspect complex amplitudes
and widths, vary PML and contour, tighten stopping tolerance and extend consecutive
checks. Store uncertainty and failed cases. Use identical physical PML collars
and monitors between candidates. Spatial discretization, geometry treatment and
reference error must be substantially below the benefit being claimed.

Optimize full angular patterns with a stable normalization; do not divide by
angular nulls. Report phase error separately. Search smooth, low-dimensional
density controls before involving a network. Charge real GPU update count,
setup/meshing time and memory; nonuniform dt and automatic stopping mean equal
cell counts need not have equal cost.

Proceed only if searched allocations consistently improve the accuracy/cost
frontier beyond reference uncertainty. A negative result is useful: improve the
mesh family, boundary treatment or search before scaling training.

## Stage 4: laptop CNN pilot

Reintroduce a small ResU-Net only after Stage 3 passes. Inputs should describe
continuous PEC shape, signed distance/boundary direction and normalized coordinates,
with electrical size and illumination conditioning. Labels come from qualified
search results. Keep the constrained projection between model and solver.

Split by base geometry before generating translations, scales or mesh variants.
Begin with a small synthetic pilot; evaluate held-out FDTD scattering and cost,
not density-map resemblance alone. Compare uniform, geometry, searched and learned
meshes. Persist configuration, geometry, mesh, method version, source/bins,
convergence policy, reference identity and failures with each example.

## Stage 5: server scaling

Rebuild native CUDA for the server GPU and reproduce the laptop numerical suite
first. Then implement sharded resumable reference/search jobs, atomic outputs,
stable seeds, measured memory limits and failure accounting. Keep generation
separate from training so stopping/reference rules cannot silently drift.
No server training, cluster scheduling or CNN training is implemented yet.
