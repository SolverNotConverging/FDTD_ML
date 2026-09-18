# Mandatory grading and density repair

Updated 2026-09-18. This supersedes the [stage-one audit](anchor_grading_stage1.md).

Every production mesh now limits adjacent widths in both directions to 1.4,
including across anchors and PML interfaces. Stricter ratios down to 1 are allowed;
grading cannot be disabled. Anchors remain exact and the total budget is unchanged.

The density suggestion defines cumulative-density quantile coordinates q[i] over
the learned interior. For a fixed budget, the deterministic mesher minimizes:

```text
sum_i abs(x[i] - q[i]) / (domain_length * number_of_free_interior_lines)
```

It jointly chooses the integer line index of every interior anchor and the line
coordinates. All min/max spacing bounds, both ratio inequalities, fixed collar
lines, boundaries, and anchors are enforced in the same mixed-integer linear
program. This replaces fixed largest-remainder allocation followed by projection.
Already legal quantiles are a zero-cost optimum and return without optimization.

The objective measures line displacement (a quantile approximation to transport
distance), not pointwise density agreement. L1 can have multiple equally good
solutions; symmetric input does not guarantee symmetric output, and spacing need
not be monotone. A ratio cap controls local jumps, not curvature. Hard constraints
can require sizeable departures from the CNN preference. The optimizer uses an
axis-normalized 1e-9 numerical width floor; other tolerances are documented in code.

## Measured cases

All examples use a 20 mm domain. The previous false-infeasibility example now
succeeds by changing allocation; it does not fall back to a uniform mesh.

| Density | Cells | Anchors (mm) | Optimized interval counts | Maximum adjacent ratio |
|---|---:|---|---|---:|
| Uniform | 40 | 9.8, 10.2 | 20 + 1 + 19 | 1.4 |
| [1, 1, 6, 6, 1, 1] | 40 | 9.8, 10.2 | 19 + 2 + 19 | 1.4 |
| [1, 100] | 10 | 10 | 2 + 8 | 1.4 |

A separate test enumerates every anchor assignment for a small two-anchor problem
and independently solves each continuous LP, verifying the global L1 optimum.
Other tests check line movement on both sides of close anchors, deterministic
repeated output, legal randomized cases, actual infeasibility, and timeout handling.

The diagnostics report normalized mean/max displacement, solver gap/status, and
meshing time per axis. Infeasible hard constraints raise `MeshInfeasibleError`;
an unfinished search or failed numerical verification raises `MeshOptimizationError`.
Large or heavily anchored problems may hit the per-axis 30-second MILP limit.
The final LP refinement and verification occur after that search limit.

## Future training penalty

`repair_loss` in `fdtdmesh.ml` compares normalized predicted density CDFs against
detached CDF targets formed by assigning equal mass to each repaired interior
cell, then integrating back into raster bins. Pass x/y collars for CPML samples.
Fixed collar mass is excluded. Multiplying an entire predicted density by a
positive scalar leaves the loss unchanged, and gradients reach the CNN density.
There is no derivative through the MILP or FDTD.

This provides the requested penalty mechanism; a training loop and trained model
are later stages. It is an auxiliary consistency loss, not an exact Boolean
feasibility penalty or electromagnetic objective. Raster rebinning can leave
nonzero loss even when the quantile mesh was already legal. Monitor the actual
projection correction as well, and tune the auxiliary weight during training.

## Reproduce

```powershell
.venv\Scripts\python.exe examples\plot_anchor_grading.py
```

Outputs PNG/SVG figures and exact coordinates/metrics in
`artifacts/anchor_grading/`. Blue shows unconstrained density quantiles and may
miss anchors; orange is the legal optimum. Purple lines are exact anchors.
Generated artifacts are ignored by Git; scripts are tracked.
