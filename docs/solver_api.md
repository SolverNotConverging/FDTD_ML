# Solver usage and Python API

This guide describes the current `tmz-conformal-ect-v1` implementation. It solves
vacuum/PEC, z-invariant TMz scattering (`Ez,Hx,Hy`) with +x plane-wave incidence.
Conformal boundaries and enlarged cells are always enabled. Geometry and mesh
preparation use the CPU; FDTD, current DFT, convergence and NF2FF use native CUDA.

## Install and open the notebooks

From the repository root, with CUDA Toolkit 13.x and the C++ build tools installed:

```powershell
uv sync --cache-dir .uv-cache --group dev --group notebooks
./scripts/build_cuda.ps1
.venv/Scripts/python.exe -m jupyterlab notebooks
```

Select the kernel belonging to this repository's `.venv`, then **Restart Kernel
and Run All Cells**. The notebooks discover the repository from either the root
directory or `notebooks/`; opening them from an unrelated working directory will fail.
For an existing editor's kernel selector, choose `.venv/Scripts/python.exe`.
See [README](../README.md) for Linux and GPU architecture build settings.

| Notebook | Contents | GPU needed? |
|---|---|---|
| [01_geometry_and_mesh](../notebooks/01_geometry_and_mesh.ipynb) | Uniform/graded grids, PML/TFSF/contour placement, exact cut faces and enlarged pairs, axis spacing | No |
| [02_gpu_scattering_and_far_field](../notebooks/02_gpu_scattering_and_far_field.ipynb) | Multi-frequency solve, stopping history, separate bin settling, absolute/dB scattering widths, complex amplitude/phase, save/load | Yes to solve; saved outputs can be viewed without one |

Notebooks include executed example outputs. Running them again updates the plots
and writes archives/figures under ignored `artifacts/notebooks/`. Geometry raster
appearance is for display only; continuous geometry determines cut coefficients.

## First solve

```python
from fdtdmesh.cases import canonical_case
from fdtdmesh.simulation import Convergence, run_scattering

case, mesh = canonical_case(
    "cylinder", ppw=24,
    frequencies=(0.9e9, 1e9, 1.1e9),
    convergence=Convergence(check_interval=256),
)
result = run_scattering(case, mesh)
assert result.converged
result.save("artifacts/cylinder.npz")
```

`canonical_case(kind="cylinder", *, ppw=24, nonuniform=False, frequency=1e9,
shift=(0,0), radius=0.47, contour_offset=1.0, pml_cells=None,
convergence=None, frequencies=()) -> (ScatteringCase, Mesh)`

`kind` is `empty`, `cylinder`, `rectangle`, `pair`, or `slot`. `ppw` is a multiple
of eight and sets each total axis budget to `6*ppw`. Geometry lengths, `shift`,
`radius`, and `contour_offset` use wavelengths at the pulse-centre `frequency`.
The returned scene and mesh use metres. The default collar is 0.5 wavelengths
thick. `nonuniform=True` uses a demonstration Gaussian density and constrained
projection; it is not an optimized geometry mesher. Projection can take tens of
seconds even when the subsequent GPU solve takes less than a second.

## Case and source configuration

Import `ScatteringCase` from `fdtdmesh.simulation`. It is a frozen dataclass;
use `dataclasses.replace(case, field=value)` to change its fields. The contained
`Scene2D` remains mutable, so construct a new scene when independent cases are needed.

| Field | Meaning / units |
|---|---|
| `scene` | Continuous `Scene2D` containing ordered PEC/vacuum primitives |
| `frequency` | Gaussian cosine pulse centre, Hz |
| `frequencies=()` | Requested positive DFT bins, Hz; empty means the pulse centre |
| `angles` | Finite nonempty 1D array, radians; default 0…359 degrees |
| `tfsf_box` | `(x0,x1,y0,y1)` in metres, enclosing the object |
| `contour_box` | Same order, enclosing TFSF, strictly outside PML |
| `pml` | `fdtdmesh.pml.PML` with fixed x/y collars |
| `source_x` | Auxiliary 1D source position, metres, between left PML and contour |
| `origin` | `(x,y)` phase reference in metres, inside TFSF; x must be anchored |
| `pulse_width_periods=1.5` | Gaussian width in pulse-centre periods |
| `pulse_delay_periods=6.0` | Pulse delay in pulse-centre periods |
| `convergence` | `Convergence` policy below |

Changing monitored bins does not change the source pulse. Bins outside its usable
bandwidth or above the temporal Nyquist limit are rejected. The source propagates
along +x; an arbitrary incidence angle is not an available parameter. The auxiliary
source is not a 2D line radiator. All lengths supplied directly to these APIs are SI.

## Geometry, mesh and PML

`Scene2D(Lx,Ly)` requires positive domain lengths. Primitive methods return the
scene, allowing chained calls:

```python
from dataclasses import replace
from fdtdmesh.scene import Scene2D
from fdtdmesh.constants import C0

base, mesh = canonical_case("empty", ppw=24)
lam = C0/base.frequency
scene = Scene2D(base.scene.Lx, base.scene.Ly)
scene.add_rectangle((2.5*lam, 3.5*lam), (2.55*lam, 3.45*lam))
scene.add_rectangle((2.87*lam, 3.13*lam), (2.85*lam, 3.5*lam), pec=False)
custom = replace(base, scene=scene)
result = run_scattering(custom, mesh)
```

Other methods are `add_circle((cx,cy),radius,pec=True)` and
`add_polygon(vertices,pec=True)` for simple, non-self-intersecting polygons.
Primitives must lie strictly inside the domain. Later primitives overwrite earlier
interiors; `pec=False` carves vacuum. PEC walls on hole boundaries are retained.
`contains(x,y)` broadcasts coordinate arrays into a PEC mask; `bounds()` and
`as_dict()` provide bounding boxes and serializable geometry.

`Mesh(x,y)` accepts strictly increasing 1D node arrays starting at zero. Cell
counts are `Nx=len(x)-1`, `Ny=len(y)-1`; node arrays are read-only. Adjacent cell
width ratios must not exceed 1.4. `mesh.diagnostics()` reports spacing and grading.

For density-based projection, import `density_mesh`, `AxisConstraints`, and
`AxisCollar` from `fdtdmesh.mesh`:

```python
from fdtdmesh.mesh import density_mesh, AxisConstraints
import numpy as np

anchors = lam*np.array([.5,.75,1,1.5,3,4.5,5,5.5])
u = (np.arange(96)+.5)/96
rho = 1 + np.exp(-((u-.5)/.18)**2)
graded = density_mesh(
    scene.Lx, scene.Ly, 144, 144, rho, rho,
    x_anchors=anchors, y_anchors=anchors,
    x_collar=custom.pml.x, y_collar=custom.pml.y,
    x_constraints=AxisConstraints(min_spacing=lam/96),
    y_constraints=AxisConstraints(min_spacing=lam/96),
    time_limit=30.0,
)
```

Density vectors describe equal-width raster bins. `density_mesh` returns exact
budgets including collar cells and accepts `x/y_anchors`, `x/y_constraints`, and
`x/y_collar`. `AxisConstraints(min_spacing=0,max_spacing=None,max_ratio=1.4)`
controls spacing. `AxisCollar(cells,thickness)` fixes symmetric collar lines.
`time_limit` applies to each axis optimizer; infeasibility and optimizer timeout
are distinct errors. For manually constructed meshes, preserve collar coordinates
using `AxisCollar.fixed_lines(length,count)`; PML validation checks exact equality.

`PML(x,y,order=3,kappa_max=3,alpha_max=0.05,R0=1e-8)` uses x/y collars and
electric-equivalent conductivity `alpha_max` in S/m. Both axes should have collars
for the documented scattering setup. Keep contour interpolation outside PML,
at least two cells between contour and TFSF, and geometry beyond the two-cell
interior guard of TFSF. Anchors must include all monitor/source coordinates.

Not every shape/grid combination has supported cut topology. Small faces need a
complete unused vacuum donor. Edges with disconnected open segments, unresolved
gaps, and hidden subcell primitives are rejected. Refinement or geometry anchors
may be necessary, especially at rotated corners. See [method](numerical_method.md).

## Running and stopping

`run_scattering(case, mesh, *, dtype="float64", safety=0.9, dt=None,
progress=None, diagnostic_download=False, require_converged=True)` returns a
`ScatteringResult`. No timestep count or fixed time window is required.

| Argument | Behavior |
|---|---|
| `dtype` | `"float32"` or `"float64"` fields; DFT accumulators remain float64 |
| `safety` | Fraction of the computed timestep bound, strictly between zero and one |
| `dt` | Optional explicit timestep in seconds, strictly below the bound |
| `progress` | Callable receiving small status dictionaries; may be coalesced |
| `diagnostic_download` | Explicit test-only field/current download after GPU NF2FF |
| `require_converged` | Default raises on failure; false returns a result to inspect |

`Convergence(max_steps=200000,check_interval=2048,stable_checks=3,
rtol=1e-5,atol=1e-8,field_tol=1e-5)` controls device stopping. `max_steps` is a
failure/resource cap (at most ten million), not the simulation duration. Counts
must be positive integers and the cap must exceed the source/propagation guard.

Each checkpoint tests changes in current and incident DFTs for every bin and the
maximum residual field. All must pass for `stable_checks` consecutive checkpoints
after the guard. The current norm uses physical contour weights and impedance-scaled
H. `atol` is scaled by incident spectral strength, so it is not an absolute SI
current threshold. Tightening these values controls transient settling, not spatial
mesh error or reference accuracy.

```python
def progress(report):
    print(report["step"], report["dft_error_ratio"], report["residual"])

result = run_scattering(case, mesh, progress=progress)
```

Reports include `generation`, `step`, integer `status`, `stable_checks`,
`simulated_time`, `dft_error_ratio`, and `residual`. They come from an asynchronous
low-priority stream; the GPU graph never waits for Python callbacks. Delivery is
throttled and short runs may show only a final callback. Callback exceptions are
raised after the autonomous GPU execution finishes.

## Result arrays and normalization

Let F be the number of frequencies and A the number of angles.

| Attribute | Shape / interpretation |
|---|---|
| `frequencies` | `(F,)`, Hz |
| `angles` | `(A,)`, radians; 0 forward, π backscatter |
| `amplitude` | `(F,A)`, complex128, dimensionless normalized far-field S |
| `width` | `(F,A)`, float64, 2D scattering width in metres |
| `history` | `(checks,8)`, checkpoint table below |
| `diagnostics` | Status, steps, timing, transfer/memory counts, grid and case configuration |
| `mesh` | The input mesh |
| `converged` | True exactly when diagnostics status is `"converged"` |
| `debug` | Empty by default; optional validation arrays |

The phasor convention is `exp(+iωt)` and outgoing radiation uses Hankel kind 2.
With `k=2πf/c0`, `width=4*abs(amplitude)**2/k`. S is normalized to the incident
DFT at the chosen phase origin. Complex phases from different origins cannot be
compared directly. To plot normalized width use `width/(C0/f)`; for a dimensionless
dB ratio use `10*log10(width/(C0/f))` with a display floor at zeros. These are
**2D scattering widths, not 3D RCS or dBsm**.

History columns, zero-based:

| Column | Value |
|---:|---|
| 0 | Update count |
| 1 | Simulated time, seconds |
| 2 | Worst-bin DFT/incident change divided by tolerance; pass below one |
| 3 | Residual field / peak incident field; pass below `field_tol` |
| 4 | Consecutive successful checks |
| 5 | Status: 0 running, 1 converged, 2 cap, 3 nonfinite |
| 6 | Peak time-domain incident field |
| 7 | Checkpoint generation |

Per-bin histories are not retained by the current multi-bin API. Notebook 02
explicitly repeats single-bin solves to visualize separate traces. It does not
present those as traces downloaded from the multi-bin solve.

`gpu_ms` measures the GPU graph, including DFT/checks/NF2FF. `setup_seconds` covers
preparation inside `run_scattering`; `wall_seconds` covers the function, including
callbacks, but excludes mesh construction done earlier. Transfer counts distinguish
final results, diagnostics and explicit debug arrays. Device byte counts account
for application buffers, not all driver/graph overhead.

With `diagnostic_download=True`, `debug` contains final `Ez(Nx+1,Ny+1)`,
`Hx(Nx+1,Ny)`, `Hy(Nx,Ny+1)`, `currents(F,Ncontour,2)` as complex `(Ez,Ht)` DFTs,
and `incident(F,Nx+1)`. This is a validation facility; ordinary visualization
of final scattering needs none of these downloads.

## Save, load and handle errors

```python
import json
import numpy as np

result.save("artifacts/result.npz")
with np.load("artifacts/result.npz", allow_pickle=False) as saved:
    amplitude = saved["amplitude"].copy()
    width = saved["width"].copy()
    history = saved["history"].copy()
    diagnostics = json.loads(str(saved["diagnostics"]))
```

The archive also contains `frequencies`, `angles`, `x`, and `y`. There is currently
no `ScatteringResult.load()` method. Loading saved results needs no GPU. Check
`diagnostics["status"]` before treating an archive as converged.

```python
from fdtdmesh.simulation import ConvergenceError
try:
    result = run_scattering(case, mesh)
except ConvergenceError as exc:
    failed = exc.result
    print(failed.diagnostics["status"], failed.diagnostics["Nt"])
    # Inspect history; do not use this result as a successful training label.
```

| Error | Typical action |
|---|---|
| CUDA unavailable / no usable device | Select project kernel, build extension, check driver/toolkit |
| `ValueError` about anchors, clearance, dt or bandwidth | Correct the case/grid/source configuration |
| `UnresolvedGeometryError` from `solver.conformal` | Resolve unsupported cut topology or donor availability |
| `MeshInfeasibleError` from `mesh` | Increase budget or revise conflicting spacing/anchor constraints |
| `MeshOptimizationError` from `mesh` | Inspect/increase projection time limit; timeout is not proven infeasibility |
| `ConvergenceError` | Inspect status and history, source excitation, residuals and safety cap |

For unusual resonators, demonstrate stability under stronger stopping policies.
For mesh comparisons, independently refine space and vary contour/PML; automatic
stopping alone does not certify scattering accuracy.

## Lower-level inspection and analytical reference

`prepare(case,mesh,dtype="float64",safety=0.9,dt=None)` validates the layout and
returns `(coefficients,tfsf_indices,source_index,contour,options)` without CUDA
stepping. Notebook 01 uses its cut lengths and pair list for inspection. Treat
these solver-internal layouts as implementation details, not a stable native API.

`fdtdmesh.scattering.cylinder_amplitude(radius,frequency,angles,incidence=0,terms=None)`
returns the independent PEC-cylinder series. `pattern_error(actual,reference)`
returns relative L2 error and rejects a zero reference norm. The CPU `far_field`
and NumPy time stepper are test oracles; production solves call `run_scattering`.
