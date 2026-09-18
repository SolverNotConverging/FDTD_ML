# Anchor grading audit

Date: 2026-09-18. This audits the existing mesher; it does not change its defaults
or allocation algorithm.

## Finding

Neighbor movement and cross-anchor grading are implemented, but grading is opt-in.
`AxisConstraints.max_ratio` defaults to `None`. Without an explicit limit,
anchor-aware allocation and quantiles can create abrupt adjacent-cell size changes.

When `max_ratio=r` is supplied, `_project` optimizes mesh-line positions subject to
anchor-interval length equalities, spacing bounds, and both inequalities for every
pair of adjacent widths:

```text
h[i+1] <= r * h[i]
h[i]   <= r * h[i+1]
```

These constraints apply across anchors as well as inside intervals. The projection
moves neighboring non-anchor lines while preserving the exact cell count and anchor
coordinates. Anchors are incorporated during mesh generation, not appended afterward.

Enable it for both axes through either density or checkpoint-based meshing:

```python
from fdtdmesh import AxisConstraints

grading = AxisConstraints(max_ratio=1.3)
sim.mesh_from_density(rho_x, rho_y, x_constraints=grading, y_constraints=grading)
# Or:
sim.mesh_with_model("mesher.pt", x_constraints=grading, y_constraints=grading)
```

The value 1.3 is an illustrative setting: the larger of two neighboring cells may
be at most 30% wider than the smaller. This audit does not establish an optimal
grading limit for electromagnetic accuracy.

## Measured examples

Domain length 20 mm, 40 cells, fixed anchors at 9.8 and 10.2 mm. Both runs retain
41 coordinate lines and both exact anchors. Grading is checked numerically on
every adjacent pair, and lines are confirmed to move on both sides of the anchors.

| Density over equally sized bins | Default maximum adjacent ratio | With r=1.30 | Lines displaced by more than 1 μm | Maximum displacement |
|---|---:|---:|---:|---:|
| Uniform | 2.578947 | 1.300000 | 14 | 0.50997 mm |
| [1, 1, 6, 6, 1, 1] | 5.869110 | 1.300000 | 24 | 0.63587 mm |

The projection minimizes squared displacement of mesh lines. It limits adjacent
ratios, but it is not a monotonic-spacing or curvature-smoothing objective. The
plots show compensating wider shoulder cells and, for the adaptive case, small
spacing undershoots. The minimum spacing can therefore decrease even while the
maximum adjacent ratio improves; use `min_spacing` too if a CFL-related floor is
required, accepting that the combined constraints may be infeasible.

## Why “graded whenever possible” is not yet guaranteed

The mesher allocates integer cell counts to anchor intervals first and keeps those
counts fixed during projection. It does not search alternative allocations after
a projection feasibility failure.

A concrete counterexample uses a 20 mm domain, 10 cells, an anchor at 10 mm,
density `[1, 100]`, and `max_ratio=1.3`:

- Current allocation: one cell in the left half and nine in the right half.
- The left cell is 10 mm wide, so its right neighbor would need to be at least
  `10/1.3 = 7.6923` mm wide and the next at least `7.6923/1.3 = 5.9172` mm.
  Those two alone cannot fit inside the 10 mm right interval. Projection rejects it.
- Five cells on each side give uniform 2 mm widths, preserve all hard constraints,
  and have adjacent ratio 1.0. This is a feasible alternative allocation, not the
  current mesher's output for the original density.

To fully implement the requested policy, enable a documented default grading
limit and make interval allocation grading-aware, or retry allocation when projection
fails. A feasible hard-constrained mesh may need to deviate substantially from the
CNN density target. Truly infeasible anchor/budget/spacing combinations must still
be reported, rather than silently dropping anchors or relaxing the grading limit.

## Reproduce the figures

Matplotlib is included in the development dependencies. From the repository root:

```powershell
uv sync --locked
.venv\Scripts\python.exe examples\plot_anchor_grading.py
```

Generated files under `artifacts/anchor_grading/`:

- `anchor_grading_comparison.png` and `.svg`: mesh-line movement, cell widths,
  and adjacent ratios for uniform and adaptive densities.
- `anchor_allocation_limit.png` and `.svg`: the failed fixed allocation and an
  independent feasible alternative satisfying the same hard constraints.
- `measurements.json`: exact coordinates, measured ratios, displacements, and
  the expected allocation error.

The generated artifacts are ignored by Git; the script and dependency lockfile
are tracked so the figures can be reproduced.
