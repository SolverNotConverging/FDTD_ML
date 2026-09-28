# Mesh optimization API

The optimization package searches exact-geometry meshes after a simulation and
reference are defined. Import public functions from `fdtdmesh.optimization`:

```python
from fdtdmesh.optimization import (
    FeasibleSettings, ReferenceSettings, SearchSettings,
    analyze_mesh_adaptivity, optimize_mesh, qualify_reference,
)
```

`optimize_mesh` keeps the physical geometry, source, frequencies, observation
angles, boundary policy, and total cell budget fixed. Native CUDA is used only
for candidate solves; projection and feasibility inspection are CPU work.

## `optimize_mesh`

```python
optimize_mesh(
    sim, reference, *, cells, directory, strategy="feasible_local",
    settings=SearchSettings(), local_settings=None, initial_mesh=None,
    constraints=None, resume=True, progress=None,
)
```

| Argument | Default | Units | Meaning |
|---|---:|---|---|
| `sim` | required | — | Source simulation; it is not modified. |
| `reference` | required | — | Qualified `Reference` target. |
| `cells` | required | cells `(Nx, Ny)` | Exact total budget, including PML. |
| `directory` | required | path | Archive root. |
| `strategy` | `"feasible_local"` | — | Current search strategy. |
| `settings` | `SearchSettings()` | — | Search budget and validation policy. |
| `local_settings` | `None` | — | `FeasibleSettings` for feasible-local proposals. |
| `initial_mesh` | `None` | — | Seed mesh; feasible-local auto-generates a geometry-aware seed when absent. |
| `constraints` | `None` | — | Optional `AxisConstraints`. |
| `resume` | `True` | — | Reuse an identical archive. |
| `progress` | `None` | callback | Receives saved trial dictionaries. |

The default feasible-local search creates a geometry-aware seed automatically
when `initial_mesh` is omitted. It reports a best feasible mesh and an optional
tighter-stop winner check; it does not establish global optimality. A reference
must be qualified before it is used as an optimization target.

### `SearchSettings`

| Field | Default | Units | Meaning |
|---|---:|---|---|
| `max_evaluations` | `80` | trials | Total candidate evaluations. |
| `max_solves` | `None` | solves | Optional solve cap. |
| `max_seconds` | `600.0` | s | Wall-time budget between evaluations. |
| `controls` | `6` | knots/axis | Proposal density controls. |
| `population` | `12` | meshes | Retained parent population. |
| `seed` | `0` | integer | RNG seed. |
| `projection_seconds` | `5.0` | s | Per-projection budget. |
| `validate_winner` | `True` | boolean | Run tighter-stop winner validation. |

### `FeasibleSettings`

| Field | Default | Units | Meaning |
|---|---:|---|---|
| `radius` | `0.75` | seed-cell widths | Initial proposal radius. |
| `min_radius` | `0.05` | seed-cell widths | Radius floor. |
| `max_radius` | `2.0` | seed-cell widths | Radius ceiling. |
| `min_movement` | `0.03` | seed-cell widths | Minimum retained displacement. |
| `repair_passes` | `3` | passes | Geometry repair passes. |
| `backtracks` | `3` | attempts | Endpoint backtracks per repair. |
| `exploration` | `0.3` | fraction | Probability of exploratory parent selection. |

## Reference qualification

```python
qualify_reference(sim, *, directory, settings=None, initial_mesh=None, progress=None)
```

### `ReferenceSettings`

| Field | Default | Units | Meaning |
|---|---:|---|---|
| `ppw` | `(48, 72, 108, 164)` | cells/λ | Refinement levels for `method="refine"`. |
| `method` | `"refine"` | — | `"refine"` or `"subdivide"`. |
| `factors` | `(2, 3, 4, 5)` | factor | Subdivision levels for `method="subdivide"`. |
| `rtol` | `0.002` | relative | RMS comparison tolerance. |
| `worst_rtol` | `0.005` | relative | Worst-frequency tolerance. |
| `max_seconds` | `900.0` | s | Qualification budget. |
| `check_boundaries` | `True` | boolean | Run temporal, PML, and contour checks. |

`method="refine"` builds increasing wavelength resolutions. `method="subdivide"`
requires an `initial_mesh` argument or an already applied `sim.mesh`; it
subdivides that seed. An unqualified reference is retained for inspection but
must not be used as an optimization target.

## Additional public helpers

`analyze_mesh_adaptivity(sim, mesh, *, constraints=None,
projection_seconds=5.0)` returns CPU-only conditional mobility bounds. Its
`constraints` default is `None`; `projection_seconds` is in seconds. These
bounds omit nonlinear geometry and donor constraints.

The configured study entry point is `examples/run_study.py examples/study.json`.
It prepares meshes only unless `--run` is supplied. Build a report from saved
results with `examples/build_report.py artifacts/study`.

`max_evaluations` includes baseline, rejected and cached trials. `max_solves`
limits unique candidate mesh evaluations, including cache hits and unsuccessful
convergence. Optional winner validation is a separately reported solve outside
these search budgets. Time limits are checked between solves. Reference temporal
checks always run; `check_boundaries=False` omits PML/contour checks and leaves the
reference unqualified. References always use strict conformal treatment, including
when the candidate simulation selects hybrid.
