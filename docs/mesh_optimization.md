# Geometry-aware mesh optimization

Mesh optimization searches the placement of grid lines while preserving one
exact continuous geometry, one source setup, and one fixed cell budget. It is a
study tool built on the public API; it does not alter geometry or claim a global
optimum.

```python
from fdtdmesh import Simulation
from fdtdmesh.optimization import SearchSettings, optimize_mesh, qualify_reference

sim = Simulation(fmin=0.9e9, fmax=1.1e9)
sim.add_circle((0.0, 0.0), 0.17, material="PEC")
reference = qualify_reference(sim, directory="artifacts/reference")
study = optimize_mesh(
    sim, reference, cells=(192, 192), directory="artifacts/optimization",
    settings=SearchSettings(max_evaluations=60, max_solves=60,
                            validate_winner=True),
)
```

Optimization defaults to hybrid boundary treatment on a private clone. Use an
explicit `boundary=BoundaryPolicy(mode="conformal")` for strict search.
The default `feasible_local` strategy automatically constructs a geometry-aware
seed when `initial_mesh` is absent. Candidate proposals preserve the seed's
budget and fixed layout constraints, then undergo exact geometry and enlarged-
cell checks. Failed proposals are recorded; they do not simplify or move the
geometry. Use `FeasibleSettings` through `local_settings` to tune proposal
radii, repair passes, backtracking, and exploration.

References support `method="refine"` and `method="subdivide"`. Subdivision
requires an initial mesh, supplied as `initial_mesh` or already applied to the
simulation. Qualification checks spatial differences and, when enabled,
PML and contour sensitivity; temporal stopping checks always run. A reference that is not qualified is
not a valid optimization target.

All public settings and defaults are listed in the [optimization API reference](mesh_optimization_api.md).
The [mesh strategy guide](mesh_strategy.md) documents exact
budgets, target spacing, caps, and construction failures. The current
notebooks include:

- `01_geometry_and_mesh.ipynb` — exact geometry and CPU mesh preparation.
- `02_gpu_scattering_and_far_field.ipynb` — one native CUDA solve and NF2FF.
- `03_hybrid_boundary.ipynb` — boundary-policy workflow and targeted checks.
- `04_geometry_aware_meshing.ipynb` — qualified references and fixed-budget optimization.
- `05_hybrid_tip_optimization.ipynb` — graded margins, persistent tip patches and hybrid search against qualified subdivided references.

Hybrid boundary behavior has a bounded accuracy screen in [validation](validation.md). Historical studies in
the [reports directory](../reports/) are preserved as evidence from their
recorded source revision and are not validation of the hybrid method.
