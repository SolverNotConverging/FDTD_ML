# Solver and meshing API

FDTDMesh solves two-dimensional, z-invariant TMz PEC scattering. The public
workflow is `Simulation` → exact geometry → `apply_mesh` → `solve` → `Result`.
Geometry coordinates, frequencies, distances, and mesh coordinates are SI
units (metres and hertz). Field updates, DFT accumulation, device stopping, and
NF2FF remain native CUDA operations.

## Construct a simulation

`Simulation` accepts keyword-only arguments:

| Argument | Default | Meaning |
|---|---|---|
| `fmin`, `fmax` | required | Positive frequency-band endpoints in Hz. |
| `solver` | `SolverSettings()` | DFT bins, stopping, precision and Courant factor. |
| `observation_angles_deg` | `None` | Observation angles in degrees; `None` uses the default full-circle sampling. |
| `domain` | `DomainPolicy()` | Physical clearances and exterior resolution. |
| `boundary` | `BoundaryPolicy()` | Strict conformal or explicit hybrid fallback. |

```python
from fdtdmesh import BoundaryPolicy, DomainPolicy, Simulation, SolverSettings

sim = Simulation(
    fmin=0.9e9, fmax=1.1e9, solver=SolverSettings(),
    domain=DomainPolicy(), boundary=BoundaryPolicy(),
    observation_angles_deg=(0, 30, 60, 90),
)
sim.add_circle((0.0, 0.0), 0.17, material="PEC")
mesh = sim.apply_mesh("geometry_aware")
result = sim.solve()
```

The automatic-domain form is the supported geometry-first construction. The
final PEC bounds determine the domain; there is no design canvas. Domain
distances are specified in wavelengths and converted using the band-centre
wavelength. Exterior cells are rounded outward once, so achieved physical gaps
are never smaller than requested. Exact geometry remains the source of truth;
it is translated internally into the zero-based solver frame.

`observation_angles_deg` is in degrees and is converted once to solver radians.
`sim.geometry` retains input coordinates, while `sim.computational_geometry`
and `sim.coordinate_offset` expose the resolved solver frame.

## Public dataclasses

### `SolverSettings`

| Field | Default | Units | Meaning |
|---|---:|---|---|
| `dft_bins` | `21` | count or Hz tuple | Uniform bins over `[fmin, fmax]`, or explicit increasing frequencies. |
| `stop` | `DFTConvergence()` | — | Device DFT stopping policy. |
| `precision` | `"float64"` | — | Field precision. |
| `courant` | `0.9` | dimensionless | CFL safety multiplier in `(0, 1)`. |

### `DFTConvergence`

| Field | Default | Units | Meaning |
|---|---:|---|---|
| `max_steps` | `200000` | time steps | Safety cap. |
| `check_interval` | `2048` | time steps | Device check interval. |
| `stable_checks` | `3` | checks | Consecutive settled checks required. |
| `rtol` | `1e-5` | relative | DFT settling tolerance. |
| `atol` | `1e-8` | normalized | DFT tolerance floor, scaled by incident amplitude and contour length. |
| `field_tol` | `1e-5` | relative | Field settling tolerance. |

### `DomainPolicy`

All distance fields below are wavelengths at the band centre; `phase_origin` is
in input metres.

| Field | Default | Units | Meaning |
|---|---:|---|---|
| `scatterer_to_tfsf_wavelengths` | `5/24` | λ | Final PEC bound to TFSF. |
| `tfsf_to_contour_wavelengths` | `4/24` | λ | TFSF to NF2FF contour. |
| `contour_to_pml_wavelengths` | `6/24` | λ | Contour to inner PML edge. |
| `pml_thickness_wavelengths` | `0.5` | λ | PML thickness. |
| `exterior_ppw` | `24.0` | cells/λmin | Exterior spacing target at `fmax`. |
| `exterior_max_spacing` | `None` | m | Explicit exterior spacing cap, replacing the `exterior_ppw` target. |
| `min_pml_cells` | `12` | cells/side | Minimum PML collar. |
| `phase_origin` | `None` | m pair | Phase origin; defaults to final PEC-box centre. |
| `margin_mesh` | `"fixed"` | — | `"graded"` lets lines inside the scatterer–TFSF margins move; their physical extents/counts and exterior axes stay fixed. |

### `BoundaryPolicy`

| Field | Default | Units | Meaning |
|---|---:|---|---|
| `mode` | `"conformal"` | — | `"conformal"` or `"hybrid"`. |
| `max_fallback_fraction` | `1.0` | fraction | Maximum staircase fallback area divided by total domain area. |
| `on_fallback` | `"warn"` | — | `"warn"` or `"error"`. |
| `max_patch_diameter` | `None` | m | Optional maximum connected fallback patch bounding-box diagonal. |

Conformal is the default. Hybrid fallback emits `StaircaseFallbackWarning` when
configured to warn. A bounded validation screen is documented in [validation](validation.md); it is not a general accuracy qualification.

### `MeshOptions`

Advanced construction controls belong in `MeshOptions`, passed as
`options=MeshOptions(...)`.

| Field | Default | Units | Meaning |
|---|---:|---|---|
| `constraints` | `None` | — | Optional `AxisConstraints`. |
| `anchors` | `None` | m pair of sequences | User-fixed feature coordinates. |
| `anchor_assignment` | `"local"` | — | `"local"`, `"joint"`, or `"fixed"`. |
| `time_limit` | `30.0` | s | Construction budget. |
| `max_passes` | `20` | passes | Construction/repair pass cap. |
| `hybrid_repair_passes` | `3` | repairs | Additional inspected hybrid proposals after the first; retain the best accepted physical patch extent when repair stalls. |

`AxisConstraints` has `min_spacing=0.0 m`, `max_spacing=None m`, and
`max_ratio=1.4` (dimensionless). Grading is constrained to at most 1.4.

| `AxisConstraints` field | Default | Units | Meaning |
|---|---:|---|---|
| `min_spacing` | `0.0` | m | Minimum adjacent cell width. |
| `max_spacing` | `None` | m | Optional maximum adjacent cell width. |
| `max_ratio` | `1.4` | dimensionless | Maximum ratio between adjacent widths. |

## Mesh strategies

| `apply_mesh` argument | Default | Meaning |
|---|---|---|
| `strategy` | `"geometry_aware"` | One of the strategies below. |
| `cells` | `None` | Exact total `(Nx, Ny)` cell budget, including exterior and PML. |
| `target_spacing` | `None` | Geometry-aware target in metres; omitted uses the builder's default. |
| `max_cells` | `(512, 512)` | Geometry-aware count cap when using a spacing target. |
| `mesh` | `None` | Existing `Mesh`, for `custom` only. |
| `density` | `None` | Positive axis-density pair, for `density` only. |
| `options` | `None` | Advanced `MeshOptions`; unavailable for `custom`. |

```python
sim.apply_mesh("uniform", cells=(144, 144))
sim.apply_mesh("quasi_uniform", cells=(144, 144))
sim.apply_mesh("geometry_aware", cells=(144, 144))
sim.apply_mesh("geometry_aware", target_spacing=sim.wavelength / 32,
               max_cells=(512, 512))
sim.apply_mesh("density", cells=(144, 144), density=(rho_x, rho_y))
sim.apply_mesh("custom", mesh=existing_mesh)
```

`uniform` requests a strictly uniform grid and may reject a fixed-layout
request. `quasi_uniform` projects the request through constraints.
`geometry_aware` accepts either an exact `cells=(Nx, Ny)` budget or a spacing
target with `max_cells`; it checks exact geometry and donor topology. `density`
projects positive axis densities at an exact budget. `custom` accepts only
`mesh=`. Cell counts include PML and all exterior cells. A failed request leaves
the previously applied mesh intact.

Adding or removing geometry invalidates the mesh and result. Apply a mesh again
before solving. See [mesh strategies](mesh_strategy.md) for the
bounded construction process and failure reports.

## Run the solver

`result = sim.solve(...)` requires an applied mesh and a CUDA GPU.

| Argument | Default | Meaning |
|---|---|---|
| `progress` | `False` | `True` prints progress; a callable receives asynchronous progress reports. |
| `diagnostic_download` | `False` | Download final fields/currents after GPU NF2FF for diagnostics. |
| `require_converged` | `True` | Raise `ConvergenceError` if stopping criteria are unmet; its `.result` contains the unqualified output. |

Ordinary solves return far fields and convergence history; field updates, current
DFTs, and NF2FF remain on the GPU. See the [numerical method](numerical_method.md)
for the stopping normalization and asynchronous reporting mechanism.

## Coordinates, archives and diagnostics

Catalog shapes use `from fdtdmesh.catalog import make_geometry` with `scale=` in
metres and `orientation_deg=` for object rotation. The source still propagates +x.
Geometry and mesh plots use input coordinates; saved configuration records the
solver translation. Observation angles are supplied in degrees, stored in radians.

`Simulation.save(path)` and `Result.save(path)` write atomic HDF5 v2 files.
`Simulation.load(path)` restores the exact design and prepares saved axes again;
`Result.load(path)` reads an immutable snapshot without executing CUDA. Boundary
policy, fallback masks/reasons, convergence, frequencies and exact geometry remain
separate. Old HDF5 v1 archives are rejected instead of silently migrated.

`sim.discretization.boundary_report` and `result.diagnostics["boundary"]` expose
fallback diagnostics. `plot_discretization(show_fallback=True, figsize=(16,14))`
shows Ez/Hx/Hy positions, enlarged pairs, exact geometry and staircase patches.
`plot_geometry(mesh=True, figsize=(16,14))` draws a large grid view. Angular
`plot_scattering` and `plot_far_field` always use polar axes.

`MeshOptions.anchors` and `anchor_assignment` apply to projected uniform/density
strategies; geometry-aware construction chooses its own repair witnesses.
Explicit `cells` cannot be combined with `target_spacing` or nondefault `max_cells`.
