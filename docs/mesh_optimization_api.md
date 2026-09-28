# Mesh optimization API reference

Import the study API from `fdtdmesh.benchmarks`. See the
[workflow guide](mesh_optimization.md) for examples, numerical qualification and
limitations, and [Notebook 03](../notebooks/03_reference_and_optimized_mesh.ipynb)
for an executable study. Arguments after `*` in signatures are keyword-only.
Lengths are SI metres, frequencies are Hz, and `incidence_deg` is in degrees.

## `optimize_mesh`

```python
optimize_mesh(sim, reference, *, cells, strategy="differential_evolution",
              directory, max_evaluations=80, max_seconds=600.0, controls=6,
              population=12, seed=0, constraints=None, feature_anchors=False,
              initial_mesh=None, local_settings=None, max_solves=None,
              projection_seconds=5.0,
              resume=True, progress=None)
```

Searches for a low-error mesh without modifying `sim`. Returns an `Optimization`.
The loss is the RMS, across frequencies, of the angular relative L2 error in
the complex far field. Each frequency has equal weight. Mesh feasibility,
convergence and the fixed cell budget remain hard constraints.

| Argument | Type | Default | Meaning and constraints |
|---|---|---|---|
| `sim` | `Simulation` | Required | Exact geometry and physical/solver configuration to evaluate. A pre-existing mesh is not required; candidates are built on fresh simulation copies. |
| `reference` | `Reference` | Required | Must have `qualified=True`, a result, and matching geometry, orientation and configuration. |
| `cells` | Pair of positive integers | Required | Exact total `(Nx, Ny)` cell counts, including PML and the full domain. |
| `strategy` | `str` | `"differential_evolution"` | `"differential_evolution"` for seeded population search, `"powell"` for bounded derivative-free local search, or `"feasible_local"` for seed-relative local proposals with exact geometry checks and restoration. The latter requires a validated geometry-aware `initial_mesh`. |
| `directory` | `str` or `Path` | Required | Root for automatically selected experiment subdirectories containing `experiment.json`, `optimization.h5`, result files and validation. |
| `max_evaluations` | Integer ≥ 2 | `80` | Total proposal trials, including previous resumed trials, failed candidates, cache hits, and baselines: three when `initial_mesh` is supplied (`geometry_aware`, `uniform`, `deterministic`), two otherwise. Changing this value selects a separate experiment. |
| `max_seconds` | Positive finite float | `600.0` | Search wall-time allowance for this invocation, in seconds. Checked between evaluations; a running solve can overrun it. Winner validation occurs afterward. |
| `controls` | Integer ≥ 2 | `6` | DE/Powell log-density knots per axis, with the last knot fixed and `2 * (controls - 1)` search variables. For `feasible_local`, the same count controls interpolation knots for local displacement proposals; it is not a fixed 10-dimensional parameterization. |
| `population` | Integer ≥ 4 | `12` | DE population members, or the retained-parent pool size for `feasible_local`; Powell does not use a population. |
| `seed` | Nonnegative integer | `0` | NumPy RNG seed for DE and `feasible_local`; Powell itself is deterministic. |
| `constraints` | `AxisConstraints` or `None` | `None` | Shared x/y spacing constraints. Without `initial_mesh`, `None` selects minimum spacing `λ/160` (reduced to half the mean fitted-object cell width when needed), maximum spacing `λ/12`, and ratio `1.4`. With `initial_mesh`, `None` selects `min_spacing=0`, `max_spacing=max(fixed exterior spacing, geometry_aware.target_spacing)`, and ratio `1.4`. |
| `feature_anchors` | `bool` | `False` | Force visible geometry corner/intersection anchors for baselines and proposals when `True`. For `feasible_local`, those coordinates must already be present in `initial_mesh`; otherwise seed validation fails. `False` retains only required layout/PML coordinates (including fitted bounding-box margins). Reference meshing keeps its own anchors and repairs. Stored in the experiment description; changing it selects a separate archive. |
| `initial_mesh` | `Mesh` or `None` | `None` | A validated `geometry_aware` mesh from the same simulation. Uses its exact axis counts and successful witness anchors as fixed constraints for all projected candidates, and scores that mesh as the first baseline. The full coordinates and anchors enter the experiment description. With `None`, the original uniform/deterministic baselines are used. |
| `local_settings` | `FeasibleSettings` or `None` | `None` | Settings for `strategy="feasible_local"`; `None` uses `FeasibleSettings()`. Supplying it to another strategy is an error. |
| `max_solves` | Positive integer or `None` | `None` | Optional cap on unique candidate FDTD evaluations, including cached hits and failed convergence, and excluding the final stricter validation. `max_evaluations` still caps proposal trials, including baselines. |
| `projection_seconds` | Positive finite float | `5.0` | Time limit for each axis mesh projection, in seconds. Projection failures remain recorded trials. |
| `resume` | `bool` | `True` | Load a matching existing archive. `False` raises `FileExistsError` if the archive already exists; it does not overwrite it. |
| `progress` | Callable accepting one dictionary, or `None` | `None` | Called after a trial checkpoint is saved. Includes trial data, one-based `number` and current `best_error`. Exceptions from callbacks propagate. |

The complete resolved experiment is written to human-readable `experiment.json`.
It includes exact geometry, domain/layout/PML, source, frequencies/angles, solver
settings, optimizer settings (including evaluation and time limits), constraints,
reference configuration/mesh/field fingerprint, and an implementation fingerprint.
The directory name is derived from this description, but reuse also compares the
actual JSON contents. Any setting difference selects a separate subdirectory;
there is no need to rename the root manually. Existing experiments remain intact.

For a geometry-seeded study, first call `baseline = sim.apply_mesh("geometry_aware")`,
then `optimize_mesh(sim, reference, cells=(baseline.Nx, baseline.Ny),
initial_mesh=baseline, ...)`. This holds the cell budget, exact geometry, exterior
coordinates, and successful repair anchors fixed; strict conformal validation can
still reject proposals that create new crossings or lose enlarged-cell donors.
The same mesh without `initial_mesh` is not an equivalent comparison.

With `strategy="feasible_local"`, proposals are generated relative to the
geometry-aware seed and its accepted local parents. The allocation of fixed
anchor indices is held constant: layout/PML lines, witness anchors, and any
requested feature anchors stay at their seed coordinates, while only free line
indices can move. Every proposal preserves the exact geometry and total cell
counts, then undergoes strict enlarged-cell and conformal checks. Geometry
failures trigger local restoration/backtracking; proposals below the
`min_movement` threshold are rejected. The report records whether raw or
repaired movement was accepted. The `radius` adapts after acceptance or
rejection, and `exploration` controls sampling of retained parents.
Here acceptance means a distinct, valid, converged mesh, even when its error
does not improve the incumbent. The radius controller is based on feasibility;
the search does not claim a formal stationarity or global-optimality certificate.
The uniform and deterministic comparison baselines retain their existing
projectors; they may use different anchor indices. The fixed-index restriction
applies to `feasible_local` search proposals, and its parent pool starts from
the geometry-aware seed. The reported best result includes all three baselines.

`max_evaluations` counts proposal trials, including the two or three baselines,
depending on whether `initial_mesh` is supplied, failed or
rejected proposals, and cache hits. `max_solves` counts unique candidate FDTD
evaluation identifiers, including cached hits and failed convergence; it does
not count the final stricter validation. The local search's conditional LP
mobility ranges are reported as bounds for the fixed seed anchor-index
allocation. They omit nonlinear geometry and donor constraints, so they are not
globally feasible mobility ranges and do not certify a minimum mesh budget.
These ranges come from an explicit `analyze_mesh_adaptivity()` call; they are
not automatically included in the optimizer report.

### `FeasibleSettings`

```python
FeasibleSettings(radius=0.75, min_radius=0.05, max_radius=2.0,
                 min_movement=0.03, repair_passes=3, backtracks=3,
                 exploration=0.3)
```

Immutable settings for `strategy="feasible_local"`. Radii and movement are in
cell-width units; repair counts are nonnegative integers.

At each free grid line the normalization is the mean of its two adjacent
**seed** cell widths. This scale stays fixed as accepted parent meshes change.
RMS movement excludes fixed lines. Search acceptance statistics exclude the
three initial baselines; internal restoration/backtracking checks are reported
separately from outer proposal trials.

| Argument | Type | Default | Meaning and constraints |
|---|---|---|---|
| `radius` | Finite float | `0.75` | Initial local proposal radius, measured in local cell-width units. Must satisfy `min_movement ≤ min_radius ≤ radius ≤ max_radius`. |
| `min_radius` | Finite float | `0.05` | Lower bound for the adaptive proposal radius. Must be at least `min_movement`. |
| `max_radius` | Finite float | `2.0` | Upper bound for the adaptive proposal radius. |
| `min_movement` | Finite float | `0.03` | Minimum maximum displacement, in cell units; smaller proposals are filtered out. Must be positive and no greater than `min_radius`. |
| `repair_passes` | Nonnegative integer | `3` | Number of local restoration attempts after a raw geometry failure. |
| `backtracks` | Nonnegative integer | `3` | Number of progressively smaller backtracking attempts after restoration attempts. |
| `exploration` | Finite float in `[0, 1]` | `0.3` | Probability of selecting a retained parent for exploration; otherwise the current best parent is used. |

### `analyze_mesh_adaptivity`

```python
analyze_mesh_adaptivity(sim, mesh, *, constraints=None,
                        projection_seconds=5.0)
```

Computes conditional LP coordinate ranges for a geometry-aware seed without
running FDTD solves. The returned ranges use the seed's fixed anchor-index
allocation and omit nonlinear geometry/donor constraints; they are not globally
feasible mobility ranges or a minimum-budget certificate. Geometry-valid
movement is assessed separately by `feasible_local` trials.

| Argument | Type | Default | Meaning and constraints |
|---|---|---|---|
| `sim` | `Simulation` | Required | Simulation whose exact geometry and layout are checked against the seed. |
| `mesh` | `Mesh` | Required | Validated geometry-aware seed mesh; its witness anchors and fixed index allocation define the conditional ranges. |
| `constraints` | `AxisConstraints` or `None` | `None` | Shared x/y spacing constraints. `None` derives seed-relative constraints from PML and geometry-aware target spacing. |
| `projection_seconds` | Positive finite float | `5.0` | Time limit in seconds for each LP projection. |

`progress` and `resume` are execution controls rather than numerical settings:
callbacks are not serialized, and `resume=False` refuses an existing exact match.
An identical interrupted experiment resumes at its checkpoint. Changing budgets
starts a distinct experiment. Legacy archives without a complete description are
preserved but not reused automatically. The returned `.directory` is the actual
selected experiment directory; use it for `Optimization.load` or `Reference.load`.
Loading a root with several experiments requires explicitly selecting one.

### Spacing constraints

Import `AxisConstraints` from `fdtdmesh.mesh` and pass it as `constraints`.
These constructor defaults differ from the wavelength-based defaults selected
when `optimize_mesh(constraints=None)` is used.

```python
AxisConstraints(min_spacing=0.0, max_spacing=None, max_ratio=1.4)
```

| Argument | Type | Default | Meaning and constraints |
|---|---|---|---|
| `min_spacing` | Finite float ≥ 0 | `0.0` | Minimum cell width in metres; coordinates must still be strictly increasing. |
| `max_spacing` | Positive finite float or `None` | `None` | Maximum cell width in metres, at least `min_spacing`; `None` adds no explicit maximum. |
| `max_ratio` | Finite float in `[1, 1.4]` | `1.4` | Maximum larger/smaller width ratio for adjacent cells. Grading cannot be disabled. |

### `Optimization` result

Obtain this object from `optimize_mesh` or `Optimization.load`; normally do not
construct it directly. Its dataclass constructor is
`Optimization(best, report, directory)`.

| Constructor argument / attribute | Type | Default | Meaning |
|---|---|---|---|
| `best` | `Result` or `None` | Required | Best feasible result, or `None` when no feasible candidate was found. |
| `report` | `dict` | Required | Trial history, best error/file/parameters, configuration, optimizer checkpoint, status and temporal validation. |
| `directory` | `Path` | Required | Base directory for relative archive paths. |
| `mesh` | `Mesh` or `None` | Read-only property | Shortcut to `best.mesh`. |

| Method | Argument | Type | Default | Meaning / return |
|---|---|---|---|---|
| `Optimization.load(directory)` | `directory` | `str` or `Path` | Required | Loads `optimization.h5` and its best result file. Returns `Optimization`; CUDA is not needed. |
| `optimization.plot_history(ax=None)` | `ax` | Matplotlib axes or `None` | `None` | Draws best feasible complex error versus proposal count. Creates axes if omitted; returns the figure. |

Useful report fields:

| Field | Meaning |
|---|---|
| `status` | `evaluation_limit`, `solve_limit`, `time_limit`, `optimizer_finished`, `no_feasible_mesh`, or `interrupted_or_failed`; `running` during work. |
| `trials` | Every proposal, including baseline kind, parameters, status, error, timing and result file when available. Failed trials have `error=None` and a message. |
| `best_error` | Best overall relative complex-field error, or `None`. |
| `validation` | Tighter-stop winner check: `passed`, error against reference, and `change` against the original winner; an unconverged check instead records a failure status/message. |
| `observed_reference_difference` | Observed refinement/sensitivity difference from the reference report; not a rigorous error bound. |
| `search_statistics` | For `feasible_local`, counts `proposals`, `raw_valid`, `returned_valid`, `solved`, `unique_solved`, `internal_checks`, and per-status `statuses`, plus `median_parent_rms_cells`, `median_seed_rms_cells`, and `median_changed_lines`. |

`MeshClearanceError` is a `MeshInfeasibleError` raised when a proposed grid lacks
the required cell clearance between geometry, TFSF, contour, source or PML.
The optimizer records that trial with `error=None` and continues searching;
the solver's clearance requirement is not relaxed. Rerun the same optimization
call to resume a saved search after updating the code.

## `ReferenceSettings`

```python
ReferenceSettings(ppw=(48, 72, 108, 164), rtol=0.002,
                  worst_rtol=0.005, max_seconds=900.0, check_boundaries=True)
```

Returns an immutable settings object for `qualify_reference`.

| Argument | Type | Default | Meaning and constraints |
|---|---|---|---|
| `ppw` | Sequence of integers | `(48, 72, 108, 164)` | At least three strictly increasing nominal points-per-wavelength levels, each ≥ 8. Axis and PML counts round to each nominal resolution while physical PML thickness remains fixed. |
| `rtol` | Positive finite float | `0.002` | Overall relative complex-field difference threshold for refinement and sensitivity checks. |
| `worst_rtol` | Positive finite float | `0.005` | Maximum permitted single-frequency relative complex-field difference. |
| `max_seconds` | Positive finite float | `900.0` | Reference qualification wall-time allowance per invocation. Checked between solves, which can overrun it. |
| `check_boundaries` | `bool` | `True` | Include thicker-PML and shifted-contour checks. `False` permits diagnostic runs but always leaves the reference unqualified for optimization. |

## `qualify_reference`

Reference grids start from a uniform target with exact geometry corner anchors.
If a thin PEC segment or vacuum gap crosses an edge with equal-material endpoints,
the generator adds an anchor inside the missed interval and reprojects the grid.
Up to eight repair passes preserve the requested axis cell counts, PML collars,
layout coordinates, and spacing constraints. The exact geometry is unchanged;
conformal topology and enlarged-cell donor validation remain mandatory. This
reference-only alignment does not alter the optimizer's fixed-budget candidates.

Every failed level is sent to `progress` with its exception type and message.
`no_valid_reference` means no level produced a valid converged result;
`spatial_unqualified` means results exist but spatial convergence was not established.
Neither status permits optimization.

```python
qualify_reference(sim, *, directory, settings=None, progress=None)
```

Returns a `Reference`, including when qualification fails. It requires two
successive passing spatial differences, then temporal and boundary sensitivity
checks. Inspect `reference.qualified` before optimization.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `sim` | `Simulation` | Required | Geometry and configuration to refine; the input is not modified. |
| `directory` | `str` or `Path` | Required | Root for per-description directories containing `experiment.json`, `reference.json`, and cached HDF5 results. All settings, including the time limit, must match for reuse. |
| `settings` | `ReferenceSettings` or `None` | `None` | `None` uses `ReferenceSettings()` defaults. |
| `progress` | Callable accepting one dictionary, or `None` | `None` | Called for successfully solved spatial levels, with `ppw`, status, difference, cache flag and time step. Failed levels and sensitivity checks are available in the final report. |

A qualified matching archive is returned immediately. An incomplete matching
study repeats its checks using cached converged solves. Changing reference
settings automatically selects a separate experiment directory.

### `Reference` result

Usually obtained from `qualify_reference` or `Reference.load`. Its dataclass
constructor is `Reference(result, report, directory)`.

| Constructor argument / attribute | Type | Default | Meaning |
|---|---|---|---|
| `result` | `Result` or `None` | Required | Latest available fine-grid result; its presence alone does not imply qualification. |
| `report` | `dict` | Required | Qualification status, spatial levels, sensitivity checks, settings and result path. |
| `directory` | `Path` | Required | Base directory for relative archive paths. |
| `qualified` | `bool` | Read-only property | Whether spatial, temporal and boundary qualification passed. |

| Method | Argument | Type | Default | Meaning / return |
|---|---|---|---|---|
| `Reference.load(directory)` | `directory` | `str` or `Path` | Required | Loads `reference.json` and its referenced HDF5 result. Returns `Reference`; CUDA is not needed. |

## `make_simulation`

```python
make_simulation(shape="circle", *, incidence_deg=0.0, frequency=1e9,
                scale=None, scale_factor=None, settings=None,
                fit_domain=True, scatterer_margin_cells=5)
```

Returns an unmeshed `Simulation` with an exact catalog shape in a fitted compact
domain (or `6λ × 6λ` when `fit_domain=False`), band `[0.9 * frequency, 1.1 * frequency]`, and 360 observation angles.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `shape` | `str` | `"circle"` | A name from `SHAPES`, listed below. |
| `incidence_deg` | Finite float | `0.0` | Incidence angle in degrees; rotates the object by its negative about the domain centre. |
| `frequency` | Positive finite float | `1e9` | Centre frequency in Hz; defines `λ = C0 / frequency`. |
| `scale` | Positive finite float or `None` | `None` | Metres per catalog recipe unit, not a multiplier or bounding-box width. With neither size argument supplied, uses `0.6λ`. Mutually exclusive with `scale_factor`. Geometry must fit the domain and solver layout. |
| `scale_factor` | Positive finite float or `None` | `None` | Dimensionless multiplier of the default `0.6λ` recipe scale: `1.5` makes the shape 50% larger. Mutually exclusive with `scale`. |
| `fit_domain` | `bool` | `True` | Fit the domain to final PEC bounds with 6 cells between PML and contour and 4 between contour and TFSF. `False` retains a fixed `6λ × 6λ` domain. |
| `scatterer_margin_cells` | Integer ≥ 3 | `5` | Exact cells per side between final PEC bounding box and TFSF when fitting. |
| `settings` | `SolverSettings` or `None` | `None` | `None` uses `SolverSettings(stop=DFTConvergence(check_interval=256))`; other solver fields retain their defaults. |

Stored observation angles are solver-frame radians. Add
`np.deg2rad(incidence_deg)` for object-frame plots.

For example, `make_simulation("wifi", scale_factor=1.5)` enlarges the default
Wi-Fi symbol by 50%. In contrast, `scale=1.5` means 1.5 metres per recipe unit;
at 1 GHz this exceeds the fixed six-wavelength domain, but the fitted domain
can grow to accommodate it. Bounds use continuous primitive extrema and
intersections with ordered PEC/air overlays, never rasterized images. Hidden
construction shapes and air cutters do not enlarge the final PEC bounding box.

## `make_geometry`

```python
make_geometry(name, *, size, scale, incidence_deg=0.0)
```

Returns an immutable continuous `Geometry`; creates no mesh and requires no GPU.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | Required | Catalog name from `SHAPES`. |
| `size` | Pair of positive floats | Required | Domain `(Lx, Ly)` in metres; geometry is centred in this domain. |
| `scale` | Positive finite float | Required | Metres per recipe unit. Preserves the recipe's relative feature dimensions. |
| `incidence_deg` | Finite float | `0.0` | Rotate geometry by minus this angle, in degrees, about its centre. |

`SHAPES` is the tuple `("circle", "ellipse", "rectangle", "triangle", "pentagon",
"convex", "l_shape", "u_shape", "star", "wifi", "moon", "sun")`.

## `run_sweep`

```python
run_sweep(shapes=SHAPES, incidences=(0.0, 30.0, 60.0, 90.0), *, directory,
          cells=(192, 192), reference_settings=None,
          strategy="differential_evolution", feature_anchors=False, max_evaluations=60,
          max_seconds=600.0, progress=None)
```

Runs shape/incidence cases serially. Returns a list of case summary dictionaries
and saves it as `study.json`. Unqualified cases remain in the list and skip
optimization. Existing matching cases resume through their individual archives.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `shapes` | Sequence of strings | `SHAPES` | Catalog shapes to run. |
| `incidences` | Sequence of finite floats | `(0.0, 30.0, 60.0, 90.0)` | Incidence angles in degrees; each shape/angle gets an independent reference and search. |
| `directory` | `str` or `Path` | Required | Root directory containing `study.json` and per-case directories. |
| `cells` | Pair of positive integers | `(192, 192)` | Total coarse-grid budget per case. |
| `reference_settings` | `ReferenceSettings` or `None` | `None` | Shared qualification settings; `None` uses defaults. |
| `strategy` | `str` | `"differential_evolution"` | Search strategy forwarded to `optimize_mesh`; also accepts `"powell"`. |
| `feature_anchors` | `bool` | `False` | Forwarded to `optimize_mesh` for every case; reference meshing is unaffected. |
| `max_evaluations` | Integer ≥ 2 | `60` | Total candidate allowance per case, including resumed trials. |
| `max_seconds` | Positive finite float | `600.0` | Search allowance per case per invocation, not the total sweep time. Reference time is configured separately. |
| `progress` | Callable accepting one dictionary, or `None` | `None` | Receives case-start, optimization-trial and case-summary events with `shape` and `incidence_deg`. |

The sweep uses `make_simulation` defaults for frequency, scale and solver settings,
and `optimize_mesh` defaults for controls, population, seed and constraints. To
customize those, loop over cases and call the lower-level APIs directly.

## `plot_gallery`

```python
plot_gallery(directory, *, shapes=None, incidences=None)
```

Returns a Matplotlib figure with shapes as rows and incidence angles as columns.
Loads saved results on the CPU; cells without a best mesh display their status.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `directory` | `str` or `Path` | Required | Sweep directory containing `study.json` and referenced case archives. |
| `shapes` | Sequence of strings or `None` | `None` | Row selection/order; `None` uses first appearance order in the manifest. Must be nonempty. |
| `incidences` | Sequence of floats or `None` | `None` | Column selection/order in degrees; `None` uses sorted unique manifest angles. Must be nonempty. |

Missing selected cases are labelled `Not run`. Geometry/mesh plots use Cartesian
axes; angular scattering and far-field plots in the notebook use polar axes.
