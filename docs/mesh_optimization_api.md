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
              initial_mesh=None,
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
| `strategy` | `str` | `"differential_evolution"` | `"differential_evolution"` for seeded population search, or `"powell"` for bounded derivative-free local search. |
| `directory` | `str` or `Path` | Required | Root for automatically selected experiment subdirectories containing `experiment.json`, `optimization.h5`, result files and validation. |
| `max_evaluations` | Integer ≥ 2 | `80` | Total proposals, including previous resumed trials, both baselines, failed candidates and cache hits. Changing this value selects a separate experiment. |
| `max_seconds` | Positive finite float | `600.0` | Search wall-time allowance for this invocation, in seconds. Checked between evaluations; a running solve can overrun it. Winner validation occurs afterward. |
| `controls` | Integer ≥ 2 | `6` | Log-density knots per axis. The last knot is fixed, leaving `2 * (controls - 1)` search variables. |
| `population` | Integer ≥ 4 | `12` | Number of DE population members. Validated and recorded for both strategies; Powell does not use a population. |
| `seed` | Nonnegative integer | `0` | NumPy RNG seed for DE. Recorded for both strategies; Powell itself is deterministic. |
| `constraints` | `AxisConstraints` or `None` | `None` | Shared x/y spacing constraints. `None` selects minimum spacing `λ/160` (reduced to half the mean fitted-object cell width when needed), maximum spacing `λ/12`, and maximum adjacent-cell ratio `1.4`, with `λ` at band centre. |
| `feature_anchors` | `bool` | `False` | Force visible geometry corner/intersection anchors for both baselines and all search proposals when `True`. `False` retains only required layout/PML coordinates (including fitted bounding-box margins). Reference meshing keeps its own anchors and repairs. Stored in the experiment description; changing it selects a separate archive. |
| `initial_mesh` | `Mesh` or `None` | `None` | A validated `geometry_aware` mesh from the same simulation. Uses its exact axis counts and successful witness anchors as fixed constraints for all projected candidates, and scores that mesh as the first baseline. The full coordinates and anchors enter the experiment description. With `None`, the original uniform/deterministic baselines are used. |
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
| `status` | `evaluation_limit`, `time_limit`, `optimizer_finished`, `no_feasible_mesh`, or `interrupted_or_failed`; `running` during work. |
| `trials` | Every proposal, including baseline kind, parameters, status, error, timing and result file when available. Failed trials have `error=None` and a message. |
| `best_error` | Best overall relative complex-field error, or `None`. |
| `validation` | Tighter-stop winner check: `passed`, error against reference, and `change` against the original winner; an unconverged check instead records a failure status/message. |
| `observed_reference_difference` | Observed refinement/sensitivity difference from the reference report; not a rigorous error bound. |

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
