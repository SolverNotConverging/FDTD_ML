# FDTDMesh — stages 1–3

A scene-first, nonuniform 2D TMz solver and a CNN-to-Yee-mesh pipeline.
The production solver runs compiled CUDA kernels through Cython. Geometry exists
independently of both the CNN raster and the simulation grid.

Project documents: [implementation plan](IMPLEMENTATION_PLAN.md),
[current progress](PROGRESS.md), [stage 2 validation](docs/stage2_validation.md),
and [grading policy and plots](docs/anchor_grading.md).

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
sim.add_PML(6, thickness=1.5e-3)
sim.add_source(
    "point", x=3e-3, y=7e-3, width=2e-11, delay=8e-11, normalization="current", amplitude=1e-3
)
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
  Monitors record **Ez only**. Use `samples=K` for K fixed physical samples on a line;
  omitted `samples` records at the mesh nodes along that line.

The default outer boundary is PEC. Materials are isotropic and nondispersive.
Geometry is point-sampled directly at each field's Yee location; this stage does
not include subpixel effective-medium averaging. Later primitives overwrite
earlier ones. PEC is an exact mask, not a large permittivity approximation.
Curved conductors still have staircase spatial error. Important features must
span several raster pixels/cells; only explicitly anchored thin lines are protected
against disappearing. Point receivers use bilinear interpolation at their exact physical coordinates;
those coordinates are returned in `result.receiver_coordinates`. Source normalization
and cross-mesh comparison conventions are described below.

## Mesh strategies

```python
sim.set_mesh(x_lines, y_lines)  # validates count, boundaries, monotonicity, anchors
sim.mesh_from_density(
    rho_x,
    rho_y,
    x_constraints=AxisConstraints(min_spacing=5e-5, max_ratio=1.4),
)
sim.load_mesh_model("models/mesher.pt", device="cuda")
sim.mesh_with_model()
# Or load and mesh together, optionally changing the budget:
sim.mesh_with_model("models/mesher.pt", Nx=80, Ny=60, device="cuda")
```

Grading is mandatory: every adjacent width pair satisfies both `h[i+1] <= 1.4*h[i]`
and `h[i] <= 1.4*h[i+1]`. `AxisConstraints` may tighten the limit into [1, 1.4];
`None` or a larger ratio is rejected. This also applies to supplied meshes.

The CNN density is a soft preference. The mesher computes its unconstrained
cumulative-density quantile lines, then **jointly optimizes anchor-to-line integer
assignments and line coordinates**, minimizing mean absolute line displacement
normalized by domain length. This defines precisely what “closest” means; it is
not a pixelwise squared-density objective. Budgets, anchors, boundaries, spacing,
and collar lines are hard constraints. Equal-quality optima may be asymmetric.

SciPy/HiGHS solves the mixed-integer linear program; an LP polishes the chosen
assignment with exact anchor bounds. A target that is already legal returns
immediately. `time_limit=30.0` controls each axis MILP search. A proven impossible
request raises `MeshInfeasibleError`; timeout or failed numerical verification
raises `MeshOptimizationError`. Constraints are never silently relaxed, and an
unfinished incumbent is never returned as an optimum. Solver tolerances apply.
The optimization uses a numerical minimum width of `1e-9 * axis_length` unless
the requested `min_spacing` is larger. Unresolvable density dynamic ranges fail.
Large budgets with many anchors can make the integer search expensive.

Mesh diagnostics include per-axis `projection_l1`, `projection_max`,
`projection_status`, `mip_gap`, and `meshing_seconds`. Corrections quantify how
much the CNN preference needs repair; they are not electromagnetic error metrics.

Thin PEC lines and finite line sources/receivers anchor their fixed coordinate and
endpoints. With no supplied mesh, `run()` uses a uniform mesh for PEC and a uniform
density preference with fixed collars for CPML. Unaligned anchors require
`mesh_from_density` or `mesh_with_model`. New anchors require remeshing.

## Fixed collars and CFS-CPML

```python
sim.add_PML(8, thickness=2e-3, direction="xy", order=3, kappa_max=3.0, alpha_max=0.05, R0=1e-8)
```

Each enabled axis reserves 8 cells **per side inside the total budget**, with
2 mm physical thickness per side. Counts/thickness can be pairs for x and y.
Omitting thickness uses the initial uniform spacing and freezes the resulting
physical thickness at configuration. Future budget changes retain these collars.
Their lines are uniform in the normal direction and fixed independently of CNN
output; grading also applies across the interior/PML interface. CNN densities
inside fixed collars do not affect interior quantiles.

PML interfaces are automatic hard anchors. Geometry, sources, and receivers must
lie strictly inside the interfaces, and nonzero interpolation support must not
enter the collars. Collars are vacuum; nonvacuum material touching PML is not yet
supported. The outermost line remains PEC behind the absorber. Staggered profiles
and four convolutional auxiliary arrays reside on the GPU and reset on every run.
`alpha_max` is electric-equivalent conductivity in S/m; `R0` sets the profile
strength and is **not a guarantee of measured reflection**. See the measured
normal/oblique/corner and late-time tests in [stage 2 validation](docs/stage2_validation.md).

## CNN contract

`fdtdmesh.ml.ResUNet` is a fully convolutional residual U-Net with FiLM conditioning
at encoder, bottleneck, and decoder scales. It produces two logits fields, followed
by stable log-mean-exp pooling over the orthogonal axis and positive softplus
densities. The deterministic mesher owns all hard mesh constraints.

Raw input channels, stored in checkpoint order:

```text
epsilon_r, sigma/(2*pi*f_max*epsilon0), PEC, source, receiver,
x_anchor, y_anchor, mu_r, PML
```

The raw `mu_r` channel exposes magnetic material physics, and the binary `PML`
channel marks fixed collars. No distance, gradient, or heuristic edge channels are used.
Global conditioning is `log(Lx*f_max/c0)`, `log(Ly*f_max/c0)`, `f_min/f_max`,
`log(Nx)`, `log(Ny)`. Spatial arrays use `(B,C,H_y,W_x)`; solver arrays use `(x,y)`.
Log-mean-exp pooling is the normalized form of log-sum-exp, making constant-density
predictions independent of raster dimensions.

Use `save_model(path, model, raster_shape=..., training_commit=..., dataset_version=...)`
and `load_model(path)`. Checkpoints include architecture, ordered channels,
normalization, raster shape, conditioning, pooling, output semantics, training
provenance, mandatory grading/objective policy, and state dictionary. The current
format is **version 2 with nine input channels**; version 1 checkpoints are rejected. Loading uses `weights_only=True` and rejects
incompatible contracts. Existing checkpoints need conversion if their architecture
or metadata differs; arbitrary old state dictionaries cannot be loaded blindly.

**No trained meshing weights are included.** `examples/cnn_mesh.py` demonstrates
the full pipeline using explicitly labelled random weights, or accepts
`--checkpoint path/to/mesher.pt`. Network gradients are supported, but integer
meshing and FDTD are deliberately not differentiable. Teacher training and
physics-generated target distillation belong to subsequent stages.

`fdtdmesh.ml.repair_loss(rho_x, rho_y, meshes, x_collars=..., y_collars=...)`
provides a differentiable auxiliary penalty against **detached** repaired-density
CDF targets. Densities have shape `(batch, axis_pixels)`. Pass the scene collars
for every CPML sample; omit them only for PEC. Equal mass per interior cell is
rebinned into CNN pixels and compared with normalized predicted mass. The loss
is scale-invariant and excludes PML-only bins. Gradients flow through densities,
not the integer optimizer or FDTD. It is a training proxy, not an exact feasibility
indicator; rebinning can introduce a residual even for a legal quantile mesh.
Use a modest weight alongside future teacher/physics objectives; correction
metrics help monitor whether predictions increasingly require less repair.

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
`dt, 2*dt, ..., Nt*dt`. Source conventions are explicit:

- Default `normalization="field_increment"` retains the original Ez increment per
  step, nearest-node placement, and right-endpoint waveform sampling. Its amplitude
  does not define a mesh-independent physical current.
- `normalization="current"` uses a point z-directed impressed current in amperes.
  Bilinear deposition weights sum to one; dividing by Ez dual-cell areas preserves
  integrated current across meshes. Samples are at half steps, and the electric
  update applies `-dt*Jz / (epsilon*(1+sigma*dt/(2*epsilon)))`.
- `normalization="current_density"` supplies Jz in A/m² at selected nodes (point or
  line). For a mesh-independent integrated point excitation, use `current`.

Sources on PEC are rejected; overlapping contributions are summed before upload.
Physical source normalization removes artificial amplitude changes with dt and
cell area, but does not remove spatial/temporal discretization error.

For mesh comparisons, use the same physical waveform, receiver coordinates, and
`t_end`. Select a shared time vector inside **all** recorded ranges and call
`result.resample(receiver, times)`; extrapolation is rejected. For spectra, use
identical frequency points and a common window/duration. `result.spectrum` uses
the unwindowed time-integral DFT of each run at its own dt; `ceil(t_end/dt)` can
make endpoints differ, so use aligned/cropped histories for strict comparisons.

During native stepping there are no Python loops/callbacks, allocations, host/device
transfers, or per-step synchronization. All coefficients, fields, source waveforms,
indices, CPML auxiliary arrays, and complete receiver histories stay on the GPU.
Bilinear receivers record four node histories per physical sample; interpolation
is applied once on the CPU after download, increasing raw history storage fourfold. Device memory uses RAII
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
Stage 2 adds global mesh-optimality checks, fixed-collar and repair-loss tests,
CPML parity in both precisions, enlarged-domain reflection controls, source-current
conservation, and physical probe/time interpolation.
CUDA tests skip when no native backend/device is available; a skipped GPU suite is
not evidence of solver validation. The reference solver is only for tests.

## Stage 3: datasets and evaluation

```powershell
.venv\Scripts\python.exe -m fdtdmesh.data generate --output artifacts/stage3/manifest.json --per-split 4 --seed 2026
.venv\Scripts\python.exe -m fdtdmesh.data evaluate --manifest artifacts/stage3/manifest.json --output artifacts/stage3/evaluation --split all --demo-cnn
.venv\Scripts\python.exe examples/plot_stage3.py --evaluation artifacts/stage3/evaluation
```

Generation creates versioned SI scene records with eight distinct splits, reproducible
seeds, physical current sources, PML configuration, raster policy, budgets and provenance.
Families include rectangles, circles, triangles, polygons, thin PEC lines, gaps,
mixed objects and touching objects. Content hashes, lineage checks and normalized
raster geometry comparisons guard against cross-split duplicates.

Evaluation refines **uniform reference grids** through 64, 128, 256 and 512 cells
per axis, retaining physical PML thickness while refining collar cells. Two
consecutive refinements must satisfy the default 2% per-receiver waveform and
spectral L2 thresholds. Failed/nonconverged references are recorded and excluded
from candidate accuracy scoring. References that cannot represent anchors uniformly
are rejected explicitly. Candidate collars retain fixed counts within their budgets.

The candidates are a uniform density preference projected onto all hard constraints,
an explicit material/edge heuristic, and an optional supplied CNN checkpoint.
`--demo-cnn` uses labelled random weights; omit it to run only the first two, or
supply `--checkpoint path.pt`. This provides no evidence of learned improvement.
Use a new empty evaluation output directory to preserve prior results.

Outputs include a copied manifest, per-scene reference histories/status, NPZ
waveforms/spectra/meshes, JSON metrics and cost diagnostics, a Markdown table, and
optional plots. All methods use common physical times, receiver coordinates,
frequencies and windowing. See [stage 3 contracts and validation](docs/stage3_validation.md)
for normalization floors, split definitions, limits and measured coverage.

## Remaining stages

Teacher training, physics-generated target distillation and demonstrated learned
improvement remain future work. Reference coverage for difficult scenes also needs
expansion: the initial 32-scene run accepted 15 references and marked 17 nonconverged.
TEz, dispersion, anisotropy, GPU batching, online DFT/energy monitors and bounded
recording remain outside the implementation. The previous FDTD library is unchanged.
