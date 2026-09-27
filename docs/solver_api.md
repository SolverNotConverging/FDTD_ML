# Solver and Python API

For reference qualification and mesh searches, see the
[optimization API reference](mesh_optimization_api.md), including argument tables.

This guide documents the public `Simulation` and `Result` workflow for
`tmz-conformal-ect-v1`: vacuum/PEC, z-invariant TMz scattering with +x plane-wave
incidence. Conformal cut faces and enlarged cells are part of the solver method.
Geometry and mesh preparation happen on the CPU; native CUDA performs field updates,
current DFT accumulation, convergence checks, and NF2FF.

For geometry-first construction, omit `size`: `Simulation(fmin=..., fmax=...)`,
add shapes in their original coordinates, then call `apply_mesh("geometry_aware")`.
See [automatic domain and geometry-aware meshing](geometry_aware_meshing.md) for
`DomainPolicy`, argument tables, coordinate transforms, limits and Notebook 04.
The explicit-domain examples below remain supported.

## Install

For the solver and native extension:

```powershell
uv sync --group dev
./scripts/build_cuda.ps1
```

Plotting requires Matplotlib. It is included in the notebook group:

```powershell
uv sync --group notebooks
```

Optional CNN inference uses ONNX Runtime:

```powershell
uv sync --extra cnn
```

No trained CNN checkpoint is supplied. The validated local platform is Windows,
Python 3.12, CUDA 13.3, MSVC, and an RTX 4070 Laptop GPU. There is no production
CPU fallback.

## Simulation object

Create one `Simulation` with a physical size and a positive analysis band. The
default `SolverSettings` uses 21 DFT bins, float64, and a 0.9 timestep safety
factor. `DFTConvergence` is an alias for the convergence policy used by the GPU.

```python
import fdtdmesh
from fdtdmesh import DFTConvergence, Simulation, SolverSettings

f0 = 1e9
lam = fdtdmesh.C0 / f0
sim = Simulation(
    size=(6 * lam, 6 * lam),
    fmin=0.9 * f0,
    fmax=1.1 * f0,
    settings=SolverSettings(
        dft_bins=21,
        stop=DFTConvergence(
            check_interval=2048,
            max_steps=200_000,
            stable_checks=3,
            rtol=1e-5,
            atol=1e-8,
            field_tol=1e-5,
        ),
        precision="float64",
        courant=0.9,
    ),
)
```

All coordinates, domain sizes, PML thicknesses, anchors, and mesh lines are SI
metres. The source is a finite Gaussian-modulated cosine. Its centre frequency is
the midpoint `(fmin+fmax)/2`. Its envelope is `exp(-((t-t0)/tau)**2)` with
`t0=5*tau` and an explicit cutoff at `10*tau`. The positive-frequency Gaussian
lobe is designed to reach -6 dB amplitude at the band edges. The real pulse has
two spectral lobes, so very broad bands have slightly different actual edge
levels; `sim.plot_source()` shows their sum. Start with 0.9–1.1 GHz for the POC.
Integer cell arguments are counts, never physical indices.

`dft_bins` may be an integer count or an explicit increasing tuple of positive Hz
frequencies. Explicit bins must lie within `[fmin, fmax]`. The source is checked for
usable excitation at every requested bin and for temporal Nyquist compliance on the
applied mesh.

Useful properties include `size`, `source`, `fmin`, `fmax`, `wavelength`,
`frequencies`, `settings`, `layout`, `pml`, `geometry`, `mesh`, and `result`.

## Exact geometry

The geometry is continuous and ordered. The public methods accept only `PEC` and
`air` materials and return a geometry handle for removal:

```python
circle = sim.add_circle((3 * lam, 3 * lam), 0.47 * lam, material="PEC")
sim.add_rectangle(
    (2.5 * lam, 3.5 * lam), (2.55 * lam, 3.45 * lam), material="PEC"
)
sim.add_polygon(
    [(2.0 * lam, 2.0 * lam), (2.3 * lam, 2.1 * lam), (2.2 * lam, 2.5 * lam)],
    material="PEC",
)
sim.add_rectangle(
    (2.87 * lam, 3.13 * lam), (2.85 * lam, 3.5 * lam), material="air"
)
sim.remove_geometry(circle)
```

The exact signatures are:

```text
add_circle(center, radius, material="PEC", name=None)
add_rectangle(x, y, material="PEC", name=None)
add_polygon(vertices, material="PEC", name=None)
remove_geometry(handle)
```

Circles are `(cx, cy), radius`; rectangles use increasing `(x0, x1)` and `(y0,
y1)` pairs; polygons use a simple, non-self-intersecting vertex sequence. The final PEC
material must lie strictly inside the domain; air cutters may extend outside it. Later primitives overlay earlier
interiors, so an `air` primitive can carve a vacuum hole in a PEC primitive while
the physical hole wall remains a PEC boundary. Touching same-material regions
merge; shared air-cut boundaries do not retain artificial PEC sheets, and an
identical air overlay completely erases the earlier PEC region. No mesh-dependent
snapping or gap closing is applied. Adding or removing geometry invalidates
the applied mesh, prepared coefficients, and previous `Simulation.result`; call
`apply_mesh` again before solving.

`sim.geometry` is an immutable continuous snapshot; `sim.discretization` is an
independent inspection copy of the mesh-dependent Yee coefficients. For CNN inputs:

```python
geometry_id = sim.geometry.geometry_id
pixels = sim.geometry.rasterize((128, 128), channels=("occupancy", "x", "y"))
sim.geometry.save("artifacts/geometry.json")
```

The raster has shape `(channels, height, width)`, sampled at pixel centres;
`x` and `y` channels are domain-normalized. Its resolution is independent of the
solver mesh. The geometry ID hashes the ordered recipe, names, and handles; it
remains unchanged by remeshing, but is not a canonical ID for geometrically
equivalent recipes. `Geometry.load` restores the standalone JSON definition.

## Applying a mesh

The public mesh strategies are:

```python
sim.apply_mesh("uniform", cells=(144, 144), strict=True)
sim.apply_mesh("deterministic", cells=(144, 144))
sim.apply_mesh("density", cells=(144, 144), density=(rho_x, rho_y))
sim.apply_mesh("cnn", cells=(144, 144), checkpoint="models/pec_mesher")
sim.apply_mesh(existing_mesh)
```

Named strategies require exact `(Nx, Ny)` total cell budgets, including fixed PML
collars and layout anchors. `uniform` first attempts exact constant spacing. If
anchors or collar interfaces cannot align, the default `strict=False` permits a
uniform density preference to go through the constrained projector. `strict=True`
is valid only for `uniform` and raises `MeshInfeasibleError` instead of accepting a
projected mesh. An explicit `Mesh` cannot be combined with `cells`, `density`,
`checkpoint`, `strict`, or other generator options.

`deterministic` uses the resolved PEC occupancy raster, smoothed boundary activity,
and occupancy to form reproducible axis densities. It is a geometry rule for
demonstrations, not a learned or accuracy-optimized baseline. `density` accepts
two finite, strictly positive one-dimensional vectors over equal-width raster bins.
Projection retains exact counts and hard anchors and enforces the adjacent-cell
width ratio policy. Proven infeasibility raises `MeshInfeasibleError`; a projection
that reaches its optimization time limit raises `MeshOptimizationError`.

The default PML uses 12 cells per side and physical thickness one half of the
layout wavelength on both axes. PML lines and interfaces are hard mesh constraints.
The TFSF and contour layout must stay outside collars and satisfy their clearance
guards. A supplied mesh is validated against those fixed lines.

### CNN checkpoint contract

`cnn` requires a directory containing `model.onnx` and `manifest.json`. The manifest
must contain the following values and fields:

```text
schema:        fdtdmesh-cnn-v1
method:        tmz-conformal-ect-v1
normalization: unit-domain
outputs:       positive-axis-density
shape:         raster shape
channels:      geometry raster channel names
sha256:        SHA-256 of model.onnx
```

The model must return `rho_x` and `rho_y` with shape `(1, bins)`. The runtime sends
the manifest channels from `Geometry.rasterize` and four physical features:
`Lx/lambda`, `Ly/lambda`, `fmin/fcentre`, and `fmax/fcentre`, where
`lambda=C0/fcentre`. The checkpoint only proposes densities. The exact-budget
mesher remains responsible for anchors, collars, spacing, and grading. ONNX Runtime
is loaded lazily and is available through `uv sync --extra cnn`; no trained model is
included in this repository.

Inputs must be named `geometry` (float32, `(1,C,H,W)`) and `physics` (float32,
`(1,4)`). `shape` is `[H,W]`; supported channels are `occupancy`, `x`, and `y`.
Inference currently uses ONNX Runtime on the CPU during mesh preparation; FDTD
and far-field conversion remain native CUDA. Density outputs must be finite and
strictly positive. Incompatible bundles fail explicitly.

## Solving and asynchronous stopping

```python
result = sim.solve(progress=True)
```

The native run does not require a caller-supplied time window. `DFTConvergence`
controls the device stop policy:

| Field | Meaning |
|---|---|
| `max_steps` | Positive safety/resource cap, at most ten million |
| `check_interval` | Steps between stopping checks |
| `stable_checks` | Consecutive passing checks required |
| `rtol` | Relative current/incident DFT change tolerance |
| `atol` | Incident-strength-scaled absolute floor |
| `field_tol` | Residual-field tolerance |

The cap must exceed the source and propagation guard. Each check evaluates current
and incident DFT changes for every requested frequency and the maximum residual
field. The device requires all criteria for `stable_checks` consecutive checks.
This is a temporal settling policy; it is not spatial convergence, PML independence,
or a general accuracy certificate.

`progress=True` prints coalesced asynchronous snapshots. A callable may be passed
instead. The callback is delivered on a low-priority telemetry path and cannot pause
or steer the GPU graph. Short runs may produce only a final snapshot. Set
`diagnostic_download=True` only when final field/current arrays are needed for
validation; ordinary far-field results do not download them.

`sim.configure_solver(dft_bins=..., stop=...)` replaces settings and invalidates
the applied mesh/preparation. On failed convergence, `solve()` raises
`ConvergenceError`; its `.result` contains the unqualified diagnostic result.
Use `require_converged=False` only to inspect such runs explicitly.

## Result data and conventions

`Result` is immutable and can be used without a CUDA runtime after the solve. Its
main fields are:

| Attribute | Meaning |
|---|---|
| `frequencies` | `(F,)` positive DFT frequencies in Hz |
| `angles` | `(A,)` observation angles in radians |
| `far_field` / `amplitude` | `(F,A)` complex dimensionless normalized amplitude |
| `scattering_width` / `width` | `(F,A)` 2D width in metres |
| `history` | checkpoint table with eight columns |
| `bin_history` | `(checks,F,3)`: current ratio, incident ratio, incident strength |
| `diagnostics` | JSON-compatible status, timing, mesh, and transfer data |
| `configuration` | JSON-compatible simulation settings and layout |
| `converged` | true only when status is `"converged"` |
| `debug` | optional arrays from `diagnostic_download=True` |

The phasor convention is `exp(+iωt)` and outgoing radiation uses Hankel functions
of the second kind. With `k=2πf/C0`,

```text
scattering_width = 4 * abs(far_field_amplitude)**2 / k
```

The width is a two-dimensional quantity in metres, not 3D RCS or dBsm. For plots,
`scattering_width/(C0/f)` is `σ₂D/λ`; the dB view is `10 log10(σ₂D/λ)` with a
display floor. Complex phase depends on the stored phase origin and should not be
compared between runs with different origins.

`result.convergence` provides named arrays: `steps`, `times`,
`current_error_ratio`, `incident_error_ratio`, `incident_strength`, `residual`,
`stable_checks`, and `status`. The two error ratios are divided by their acceptance
tolerances. `plot_convergence` plots their maximum for each bin. The per-bin
history has a 64 MiB allocation cap; increase checkpoint spacing if needed.

## Saving, loading, and plotting

Simulation definitions and completed results use versioned HDF5 archives:

```python
sim.save("artifacts/cylinder/simulation.h5")
sim2 = Simulation.load("artifacts/cylinder/simulation.h5")

result.save("artifacts/cylinder/result.h5")
loaded = fdtdmesh.Result.load("artifacts/cylinder/result.h5")
```

`Simulation.save` stores the continuous definition and optional applied mesh.
`Result.save` stores geometry, mesh, configuration, far-field arrays, widths,
convergence histories, and diagnostics. Loaded results are plot-ready without a
CUDA runtime.

Archive schema `fdtdmesh-h5-v1` contains `/config/json`, `/geometry/recipe`,
`/mesh/{x,y,metadata}`, `/far_field/{frequencies,angles,amplitude,scattering_width}`,
`/convergence/{checkpoints,current_error_ratio,incident_error_ratio,incident_strength}`,
`/source/json`, and `/diagnostics/json` (result-only groups omitted for simulation
archives). Geometry and mesh hashes are checked on load. Complex amplitudes are
stored as complex128; no equivalent-current arrays are archived by default.

Every plot method returns a Matplotlib `Figure` and accepts an optional axes object
where applicable:

```python
sim.plot_geometry(mesh=True)
sim.plot_mesh()
sim.plot_discretization()
sim.plot_source()

loaded.plot_geometry(mesh=True)
loaded.plot_mesh()
loaded.plot_convergence(per_bin=True)
loaded.plot_scattering(normalize="wavelength")
loaded.plot_far_field(component="phase")
```

`Result.save_plots(directory)` writes geometry, mesh, convergence, scattering,
amplitude, and phase PNGs. Matplotlib is not required for native stepping; install
the notebook group before calling plot methods.

Geometry/mesh plots accept `units="m"` or `"wavelength"`; the latter uses the
pulse-centre wavelength. `plot_source(ax=...)` takes two axes. Scattering plots
accept `scale="linear"`/`"db"` and `normalize="wavelength"`/`"metres"`.
All angle-dependent plots use polar axes, with zero degrees at +x and angles
increasing counterclockwise. When supplying `ax`, create it with
`plt.subplots(subplot_kw={"projection": "polar"})`. Signed real/imaginary values,
dB levels, and phase retain their signed radial labels; phase ranges from −π to π.
The `complex` view plots Re S against Im S on Cartesian axes because its axes
are complex components, not scattering angle.
Far-field components are `magnitude`, `real`, `imag`, `phase`, or `complex`.
Frequency selection must match a stored bin; omission selects the middle bin.

## Errors and compatibility

Typical errors identify the corrective action:

| Error | Meaning |
|---|---|
| `MeshInfeasibleError` | Hard budget, collar, anchor, spacing, or strict-uniform conflict |
| `MeshOptimizationError` | Density projection did not finish or failed final verification |
| `UnresolvedGeometryError` | Cut topology or enlarged-cell donor is unsupported |
| `ConvergenceError` | Device stop reached a cap/nonfinite state before qualification |
| `ValueError` | Invalid SI geometry, bins, layout, timestep, or source bandwidth |

For unusual geometries, refine independently, vary contour/PML placement, and retain
the stopping history. Automatic temporal stopping does not replace spatial or
boundary-sensitivity checks.

The public `Simulation`/`Result` classes are the stable user-facing layer. The
lower-level `ScatteringCase`, `run_scattering`, `prepare`, native runtime, and mesh
projection functions remain useful for tests and specialized numerical work, but
their internal layout and signatures are less stable. `fdtdmesh.cases.canonical_case`
is a small benchmark helper rather than the primary API.

## Compact domain and cell allocation

After defining all geometry, call:

```python
sim.fit_domain(scatterer_margin_cells=5, exterior_cells=(6, 4))
sim.apply_mesh("deterministic", cells=(192, 192))
```

| Argument | Default | Meaning |
|---|---|---|
| `scatterer_margin_cells` | `5` | Integer ≥ 3; cells from each side of the final PEC bounding box to TFSF. |
| `exterior_cells` | `(6, 4)` | Cells from inner PML edge to contour, then contour to TFSF; each count ≥ 2. |

`fit_domain` returns the simulation. It resizes the domain and translates the
whole exact recipe and phase origin together, retaining their relative position.
It invalidates the previous mesh/result. Physical gap widths use each axis's
PML cell width (wavelength/24 for an axis without PML). PML thickness/counts are
unchanged. The translated phase origin must remain inside the fitted TFSF box.
Adding geometry afterward may require fitting again.

With 12 PML cells per side and 192 cells per axis, the layout uses 24 PML,
12 PML-to-contour, 8 contour-to-TFSF, and 10 scatterer-margin cells. This leaves
138 cells across the scatterer bounding box per axis. All strategies use the
same reserved coordinates. Insufficient budgets or incompatible spacing bounds
raise a mesh-infeasibility error rather than relaxing the counts or grading.

`ScatteringLayout` also accepts `exterior_cells=None` and
`scatterer_margin_cells=None` for custom layouts without reserved counts.
Default `Simulation` layouts reserve 6/4 exterior cells; `fit_domain` adds the
scatterer margin and removes excess domain space. Fine study references turn
these count constraints off and refine the whole domain at fixed physical boxes.
