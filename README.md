# FDTDMesh — stage 1

A scene-first, nonuniform 2D TMz solver and a CNN-to-Yee-mesh pipeline.
The production solver runs compiled CUDA kernels through Cython. Geometry exists
independently of both the CNN raster and the simulation grid.

Project documents: [implementation plan](IMPLEMENTATION_PLAN.md),
[current progress](PROGRESS.md), and [validation results](docs/validation.md).

## Install and build (Windows)

Requirements: Python 3.11+, `uv`, Visual Studio C++ x64 tools and Windows SDK,
CUDA toolkit with `nvcc`, and an NVIDIA GPU. Development was validated with
Python 3.12, CUDA 13.3, MSVC 14.51, and an RTX 4070 Laptop GPU.

From the repository root in PowerShell:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
uv sync --python 3.12
.\scripts\build_cuda.ps1
.venv\Scripts\python.exe examples\simple_uniform.py
.venv\Scripts\python.exe examples\simple_nonuniform.py
.venv\Scripts\python.exe examples\cnn_mesh.py
```

The environment includes CUDA-enabled PyTorch from the explicit `cu130` index.
PyTorch provides the CNN; the FDTD solver uses its own CUDA runtime, not PyTorch
operations. Rebuild the extension after changing Python versions or native sources.
The build script discovers Visual Studio, including Insiders installations.
The extension targets compute capabilities 7.5 and 8.9 and includes 8.9 PTX.
On Linux, activate the environment and use
`FDTDMESH_BUILD_CUDA=1 python setup.py build_ext --inplace` with `nvcc` on PATH;
that build path is provided but has not been validated here.

## Simulation API

```python
from fdtdmesh import FDTD_2D_Ez, AxisConstraints

sim = FDTD_2D_Ez(
    x_range=20e-3,
    y_range=15e-3,
    Nx=80,
    Ny=60,
    f_max=20e9,
    t_end=1e-9,
)
glass = sim.add_material("glass", epsilon_r=4.0, sigma_e=0.01)
sim.add_rectangle(glass, x_position=(7e-3, 12e-3), y_position=(3e-3, 12e-3))
sim.add_source("point", x=3e-3, y=7e-3, width=2e-11, delay=8e-11)
rx = sim.add_receiver("point", x=16e-3, y=7e-3)
sim.mesh_uniform()
result = sim.run()
trace = result.receivers[rx][:, 0]
print(result.diagnostics)
```

All geometry, source, receiver, and anchor positions are **physical metres**,
including integer arguments. `Nx, Ny` are exact cell counts, so the grid contains
`Nx+1, Ny+1` coordinate lines. This preserves the material/geometry/source style
of the original local `FDTD` library, but intentionally changes its mesh lifecycle.

Supported scene operations:

- `add_material(name, epsilon_r=1, mu_r=1, sigma_e=0)`; predefined vacuum and PEC.
- `add_rectangle`, `add_circle`, `add_triangle`, `add_polygon` (simple polygons).
- `add_pec_line(x=scalar, y=(start, stop))`, or the horizontal equivalent.
- `add_anchor("x" | "y", position)`.
- `add_source("point" | "line" | "line-soft", x=..., y=...)`.
- Gaussian, Gaussian-modulated sine, and sine/CW sources, with amplitude, frequency,
  width, delay, and phase parameters. Gaussian defaults are width `1/f_max`,
  delay `4*width`; choose a duration long enough to observe the pulse.
- `add_receiver("point" | "line", ...)` and `add_line_monitor(x=..., y=...)`.
  Monitors record **Ez only**, with one column per sampled node.

The default outer boundary is PEC. Materials are isotropic and nondispersive.
Geometry is point-sampled directly at each field's Yee location; this stage does
not include subpixel effective-medium averaging. Later primitives overwrite
earlier ones. PEC is an exact mask, not a large permittivity approximation.
Curved conductors still have staircase spatial error. Important features must
span several raster pixels/cells; only explicitly anchored thin lines are protected
against disappearing. Point probes snap to the nearest Ez node; actual sampling
coordinates are returned in `result.receiver_coordinates`.

## Mesh strategies

```python
sim.set_mesh(x_lines, y_lines)  # validates count, boundaries, monotonicity, anchors
sim.mesh_from_density(
    rho_x,
    rho_y,
    x_constraints=AxisConstraints(min_spacing=5e-5, max_ratio=1.5),
)
sim.load_mesh_model("models/mesher.pt", device="cuda")
sim.mesh_with_model()
# Or load and mesh together, optionally changing the budget:
sim.mesh_with_model("models/mesher.pt", Nx=80, Ny=60, device="cuda")
```

The deterministic mesher integrates positive, piecewise-constant axis densities,
allocates cells to anchor intervals by capped largest remainder with stable ties,
and places exact cumulative-density quantiles. Optional constraints use a linear
feasibility solve and a quadratic line-position projection with fixed anchors.
Everything is normalized by physical length during optimization.

`min_spacing`, `max_spacing`, and `max_ratio` apply to the entire axis, including
across anchors. Infeasible requests raise `MeshInfeasibleError`. Grading projection
holds the integer interval allocation fixed; it can reject an allocation even if
a different allocation could satisfy the constraints. It never drops an anchor or
silently relaxes a constraint. Anchors closer than float64 resolution are rejected.

Thin PEC lines and finite line sources/receivers anchor their fixed coordinate and
endpoints. Calling `run()` without a mesh uses a uniform grid. A uniform grid whose
lines cannot represent every anchor raises an error; use `mesh_from_density` or
`mesh_with_model` in that case. Adding new anchors after meshing requires remeshing.

## CNN contract

`fdtdmesh.ml.ResUNet` is a fully convolutional residual U-Net with FiLM conditioning
at encoder, bottleneck, and decoder scales. It produces two logits fields, followed
by stable log-mean-exp pooling over the orthogonal axis and positive softplus
densities. The deterministic mesher owns all hard mesh constraints.

Raw input channels, stored in checkpoint order:

```text
epsilon_r, sigma/(2*pi*f_max*epsilon0), PEC, source, receiver,
x_anchor, y_anchor, mu_r
```

The additional raw `mu_r` channel makes the optional magnetic material physics
visible to the network. No distance, gradient, or heuristic edge channels are used.
Global conditioning is `log(Lx*f_max/c0)`, `log(Ly*f_max/c0)`, `f_min/f_max`,
`log(Nx)`, `log(Ny)`. Spatial arrays use `(B,C,H_y,W_x)`; solver arrays use `(x,y)`.
Log-mean-exp pooling is the normalized form of log-sum-exp, making constant-density
predictions independent of raster dimensions.

Use `save_model(path, model, raster_shape=..., training_commit=..., dataset_version=...)`
and `load_model(path)`. Checkpoints include architecture, ordered channels,
normalization, raster shape, conditioning, pooling, output semantics, training
provenance, and state dictionary. Loading uses `weights_only=True` and rejects
incompatible contracts. Existing checkpoints need conversion if their architecture
or metadata differs; arbitrary old state dictionaries cannot be loaded blindly.

**No trained meshing weights are included.** `examples/cnn_mesh.py` demonstrates
the full pipeline using explicitly labelled random weights, or accepts
`--checkpoint path/to/mesher.pt`. Network gradients are supported, but integer
meshing and FDTD are deliberately not differentiable. Teacher training and
physics-generated target distillation belong to subsequent stages.

## Numerical and runtime contract

Fields: `Ez (Nx+1,Ny+1)`, `Hx (Nx+1,Ny)`, `Hy (Nx,Ny+1)`.
Magnetic updates use primal cell widths; electric updates use arithmetic-average
dual widths. Electric conductivity uses the centered lossy update.

The conservative CFL bound uses minimum x/y spacings and a wave-speed upper bound
from separate minimum sampled epsilon and mu. Default `dt=0.95*dt_CFL`; supplied
`dt` must be strictly below the bound. `t_end` gives `Nt=ceil(t_end/dt)`;
`Nt` and `t_end` are mutually exclusive. Compare meshes at the same **physical
duration**, reporting `Nx*Ny*Nt` and GPU time as well as electromagnetic error.

`run()` starts from zero fields unless `initial_fields` is supplied. Initial Ez is
at time zero and Hx/Hy at `-dt/2`. Receivers are sampled after source injection at
`dt, 2*dt, ..., Nt*dt`. Soft sources add the waveform directly to Ez (V/m per step),
not an impressed current density; normalize an excitation explicitly when comparing
different temporal discretizations. Sources on PEC nodes are rejected; overlapping
sources are summed deterministically before upload.

During native stepping there are no Python loops/callbacks, allocations, host/device
transfers, or per-step synchronization. All coefficients, fields, source waveforms,
indices, and complete receiver histories stay on the GPU. Device memory uses RAII
cleanup, including error paths. CUDA event time excludes setup/transfers; total wall
time includes preparation and transfers. Transfer byte counts and a stepping-transfer
counter are returned for auditing. Full waveforms and histories consume memory
proportional to `Nt`; batching and bounded-history recording are future work.

Returned fields are the final state. `result.spectrum(receiver, frequencies)`
computes a time-integral DFT after the run on the CPU; there is no in-loop spectral
accumulator in this stage. There is no automatic CPU solver fallback.

## Tests

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
.venv\Scripts\python.exe -m pytest -q --basetemp=artifacts/pytest
.venv\Scripts\ruff.exe check .
```

Tests cover deterministic mesh budgets, anchors, quantiles, constraints, material
and PEC geometry, scale conditioning, checkpoint compatibility, CNN gradients,
uniform/nonuniform CUDA agreement with a NumPy oracle, analytic cavity convergence,
PEC shielding, runtime residency accounting, repeated runs, and CNN-to-CUDA integration.
CUDA tests skip when no native backend/device is available; a skipped GPU suite is
not evidence of solver validation. The reference solver is only for tests.

## Stage boundary

This stage implements the PEC CUDA TMz solver and CNN-to-FDTD mesher. CPML,
fixed PML collars, TEz, dispersion, anisotropy, GPU batching, online DFT/energy
monitors, dataset generation, teacher training, and physics optimization are not
implemented. `add_PML()` explicitly raises rather than pretending to absorb waves.

The source is newly implemented; the previous library is used as an API/numerical
reference and is not modified. See `docs/validation.md` for measured validation.
