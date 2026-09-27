# Reference-qualified mesh optimization

See the [optimization API reference](mesh_optimization_api.md) for signatures,
argument tables, defaults, return values and resume behavior for every public
study function.

Open `notebooks/03_reference_and_optimized_mesh.ipynb` for the complete workflow.
It runs the selected shape and incidence on the GPU. The optional catalog sweep covers
12 shapes at 0, 30, 60 and 90 degrees; enable it for longer server experiments.
All lengths are metres and public geometry rotation angles are radians.

Notebook 03 accepts `CELLS = [(64, 64), (96, 96), (128, 128)]`, or a single
tuple for one budget. It qualifies one shared reference, runs a separate search
per budget, and stores results in `optimizations[(Nx, Ny)]`. Search limits apply
per budget. Each budget gets large mesh/discretization figures, error history,
and polar field comparisons; a search without a feasible mesh gets an explicit
status panel. Repeated identical budgets are run once. The optional catalog
sweep has its own separately configured cell budget.

Notebook 03 uses strict conformal validation throughout. The reference simulation
uses exact geometry feature anchors and local edge-alignment repairs. Optimization
uses `feature_anchors=False` while preserving the exact geometry and validating all
candidate topology and enlarged-cell requirements. Other infeasible configurations
still fail.

For relative resizing, use `make_simulation(..., scale_factor=1.5)` (50% larger).
`scale=` remains an absolute SI length per recipe unit. Notebook 03 shares one
`SCALE_FACTOR` between its previews, selected simulation and mesh inspection.

The benchmark now fits its domain to the final continuous PEC bounding box. Each
side has 5 cells to TFSF, 4 more to the integration contour, 6 more to the inner
PML edge, then 12 PML cells. At `(192, 192)`, this leaves 138 cells per axis
across the scatterer's bounding box. Reserved axis coordinates are identical
across candidate strategies. Tensor-product lines still extend tangentially
through the exterior; this does not create a separate block mesh.

For a manually constructed simulation, call
`sim.fit_domain(scatterer_margin_cells=5, exterior_cells=(6, 4))` after adding
geometry and before meshing. It translates the full recipe and phase origin
together and invalidates an existing mesh. Gap widths use the adjoining PML
cell width. Fine references preserve these physical boxes but refine all gaps
too. Old reference/search caches are preserved; complete JSON descriptions select
compatible experiments automatically.

```python
from fdtdmesh.benchmarks import (
    make_simulation, ReferenceSettings, qualify_reference, optimize_mesh,
)

sim = make_simulation("moon", incidence_deg=30, frequency=1e9)
reference = qualify_reference(
    sim, directory="artifacts/moon_30/reference",
    settings=ReferenceSettings(ppw=(48, 72, 108, 164)),
)
if reference.qualified:
    study = optimize_mesh(
        sim, reference, cells=(192, 192), strategy="differential_evolution",
        directory="artifacts/moon_30/optimization",
        max_evaluations=60, max_seconds=600, seed=0,
    )
    if study.best is not None:
        study.best.plot_geometry(mesh=True, units="wavelength")
        study.best.plot_scattering()
        study.plot_history()
```

The same APIs accept a user-created `Simulation`. Its geometry, source, angular
samples, frequency band, and solver settings identify a reference case. Changing
them automatically selects a separate experiment subdirectory. There is no geometry rasterization in
the optimizer: analytic circles/ellipses and polygon vertices remain fixed.
`Simulation.add_ellipse(center, radii, angle=0, material="PEC")` adds an exact
ellipse. `Geometry.rotated(angle, origin=...)` returns a new continuous recipe;
`Simulation.set_geometry(geometry)` installs it and invalidates the old mesh.

## Reference qualification

References refine the entire domain and PML at increasing nominal points per
wavelength. Polygon corners and visible circle/polygon intersections anchor
grid lines; consequently these are uniform-target, feature-aligned meshes, not
necessarily strictly uniform meshes. Qualification requires two consecutive
refinement differences below both the overall and worst-frequency tolerances,
then tighter DFT stopping, thicker PML, and a shifted NF2FF contour checks.
Contour checks preserve the mesh and choose existing grid lines. Physical PML
thickness stays fixed during spatial refinement; the separate PML check expands
the domain and absorber while preserving interior coordinates.

Reference axis and PML cell counts are rounded to the nominal resolution while
physical PML thickness remains fixed. `reference.report` records all attempted levels,
failures, sensitivity checks, and an observed reference difference. This is a
numerical reference, not an exact solution or a rigorous error bound. Compare
claimed improvements with those observed differences. Unqualified references
cannot be passed to `optimize_mesh`. Increase the resolution/time budget or
inspect unresolved topology instead of treating failed cases as valid targets.

## Search strategies and budgets

`strategy="differential_evolution"` uses a seeded rand/1/bin population search.
`strategy="powell"` uses bounded SciPy Powell search. Both are derivative-free:
the native CUDA solver, changing cut topology and integer anchor assignments
do not currently provide gradients for Torch autograd.

The search adjusts positive axis densities using `controls` log-density knots
per axis. The final knot fixes an otherwise redundant density scale. A mesh
projection enforces exact `(Nx, Ny)` total cell counts, PML collars, layout and
optional geometry anchors, spacing bounds, and adjacent-cell grading. The default minimum
spacing is the smaller of wavelength/160 and half the mean spacing needed inside
the fitted scatterer bounding box; maximum spacing is wavelength/12. Supply `AxisConstraints`
to change these. Cell count includes PML and the entire domain.

Optimization defaults to `feature_anchors=False`: sharp corners do not force grid
lines. Set `feature_anchors=True` to restore corner/intersection anchoring for
both baselines and search candidates. Reference anchoring and local repairs remain
independent. The deterministic density seed still guides proposals using geometry.
Conformal topology and enlarged-cell donors are always validated; invalid
candidates are recorded and the search continues. A `no_feasible_mesh` result
means no valid candidate was found within this search and budget, not proof that
no valid mesh exists. Geometry is never simplified to make a candidate feasible.

Study meshes use `anchor_assignment="local"`: integer anchor indices can move
within two indices of their uniform positions. This bounds projection cost but
restricts the search space. Infeasibility means infeasibility under these search
constraints, not proof that no possible mesh exists. General `apply_mesh` also
accepts `anchors=(x_coordinates, y_coordinates)` and `anchor_assignment="joint"`
(default, full integer search) or `"fixed"` (fixed nearest feasible indices).
Feature extraction handles the catalog's polygon/circle intersections; arbitrary
CSG involving ellipse intersections may require user-supplied anchors.

Both uniform-target and deterministic baselines count toward `max_evaluations`.
Failed proposals and cache hits count too. Known geometry, convergence and mesh
failures remain in the trial archive; unexpected software errors propagate.
The winner is the best feasible mesh found, including baselines. Improvement and
global optimality are not guaranteed. A separate tighter-stop winner check is
recorded under `report["validation"]`; inspect its `passed` flag before using
the result. This check runs after the search budget.

`max_seconds` limits each invocation between evaluations; a running projection
or CUDA solve can overrun it. `projection_seconds` caps each axis projection.
The native CFL time step varies by mesh, so errors measure end-to-end solver
accuracy and include temporal error. This is not a spatial-only fixed-dt study.
The reserved exterior axis coordinates are fixed across candidates; tangential
lines still change because the solver uses a tensor-product grid.

## Archives, resume and sweeps

Solver results remain HDF5. `optimization.h5` contains the report, trial records,
population and RNG state; its referenced result HDF5 files contain exact geometry,
mesh, complex fields, widths and DFT histories. `reference.json` and `study.json`
are small status manifests. Keep each case directory together when moving it.
Only converged solves are cached. Each experiment and solver result has an
`experiment.json` containing the complete resolved configuration; solver manifests
also include the actual mesh coordinates. Reuse requires exact JSON equality.
Mismatches automatically select separate subdirectories under the supplied root,
preserving prior runs. The returned `.directory` identifies the selected archive.
Old archives without a complete JSON description are not reused automatically.

Repeat an identical call to resume. Changing `max_evaluations`, `max_seconds` or
any other numerical setting starts a separate experiment. DE resumes its population at saved evaluation boundaries; Powell
warm-restarts from the best available parameters. A process killed during an
evaluation restarts from its last saved checkpoint. `Optimization.load(directory)`
and `Reference.load(directory)` load saved results without CUDA.

```powershell
.venv/Scripts/python examples/optimize_meshes.py --shapes circle ellipse wifi moon sun --incidences 0 30 60 90 --cells 192 192 --evaluations 60 --output artifacts/catalog
```

`run_sweep(..., directory=...)` runs cases serially and records unqualified cases
explicitly. `plot_gallery(directory)` displays the saved best mesh per shape and
incidence, or a status label. The source remains +x: incidence experiments rotate
the object by minus the requested angle around the common phase origin. Add that
angle to stored solver observation angles for plots in the original object frame.
These are rotated-object experiments, not arbitrary-angle TFSF injection.

## Earlier-layout verification

The following historical results predate the compact 5/4/6 layout.

GPU pilots used the default band and geometry scale, reference levels
48/72/108/164, and 192 by 192 candidate cells. A short DE search with four
controls per axis, population four, seed zero and 12 total evaluations produced:

| Shape / incidence | Uniform-target complex error | Best found complex error |
|---|---:|---:|
| Rectangle / 0 degrees | 1.346% | 1.217% |
| Ellipse / 30 degrees | 1.578% | 1.219% |
| Wi-Fi / 0 degrees | 1.693% | 1.437% |
| Moon / 0 degrees | 1.679% | 1.380% |

These references qualified and the winners passed tighter temporal stopping.
The rectangle improvement is below its observed reference difference; the table
is search evidence, not proof of a resolved improvement in every case. The
executed notebook uses its own 60-evaluation circle example. The sun at 90 degrees
remained spatially unqualified at these reference levels. The complete 48-case
catalog sweep has not been run; more difficult corners may need finer reference
levels, larger coarse budgets, or different spacing constraints.
