# PEC scattering and nonuniform meshing

FDTDMesh is a research solver for two-dimensional, z-invariant TMz scattering from
continuous PEC geometry. It uses a tensor-product nonuniform Yee mesh, conformal
cut-face coefficients, enlarged cells where a cut face needs a donor, a +x
total-field/scattered-field (TFSF) incident wave, GPU current DFT accumulation,
and GPU near-to-far-field (NF2FF) conversion. The reported quantity is a 2D
scattering width in metres; it is not a 3D radar cross section.

The supported user workflow is:

```text
Simulation -> add exact geometry -> apply_mesh -> solve -> Result
```

Geometry and mesh preparation run on the CPU. Native CUDA performs field updates,
DFT accumulation, convergence checks, and NF2FF. The device controls the stopping
decision. Progress telemetry is asynchronous and does not control the stepping loop.

The active project contains the clean PEC restart. The previous dielectric/CNN
experiments are preserved on the `legacy-project` branch and are not part of this
runtime or dependency set.

## Install and build

Python 3.11+, an NVIDIA GPU, CUDA Toolkit 13.x, a compatible driver, and a supported
C++ compiler are required. The validated local environment uses Python 3.12,
CUDA 13.3, MSVC, and an RTX 4070 Laptop GPU. There is no production CPU fallback.

```powershell
uv sync --group dev
./scripts/build_cuda.ps1
.venv/Scripts/python.exe -m pytest -q
```

The default native target is SM 89. Set `FDTDMESH_CUDA_ARCHS` before building on
another GPU, for example `80,89`; the final entry also receives PTX. Rebuild after
changing native sources or the target architecture.

The intended Linux build path is available but has not been qualified here:

```bash
uv sync --group dev
FDTDMESH_BUILD_CUDA=1 FDTDMESH_CUDA_ARCHS=80 uv run --no-sync python setup.py build_ext --inplace
uv run --no-sync pytest -q
```

## A first solve

All lengths and coordinates are SI metres. The default source is a finite Gaussian-
modulated cosine pulse. `fmin` and `fmax` define its approximately -6 dB edge frequencies; the
default pulse cutoff is five Gaussian widths. The default analysis contains 21 DFT
bins over that band.

```python
import fdtdmesh
from fdtdmesh import DFTConvergence, Simulation, SolverSettings

f0 = 1e9
lam = fdtdmesh.C0 / f0
settings = SolverSettings(
    dft_bins=21,
    stop=DFTConvergence(check_interval=2048, max_steps=200_000, rtol=1e-5),
    precision="float64",
)
sim = Simulation(
    size=(6 * lam, 6 * lam),
    fmin=0.9 * f0,
    fmax=1.1 * f0,
    settings=settings,
)
sim.add_circle((3 * lam, 3 * lam), 0.47 * lam, material="PEC")
sim.apply_mesh("uniform", cells=(144, 144), strict=True)
result = sim.solve(progress=True)
result.save("artifacts/cylinder/scattering.h5")
result.save_plots("artifacts/cylinder")
```

The `examples/pec_scattering.py` command exposes the same workflow for the
`empty`, `cylinder`, `rectangle`, `pair`, and `slot` canonical shapes:

```powershell
.venv/Scripts/python.exe examples/pec_scattering.py --shape cylinder --ppw 24
.venv/Scripts/python.exe examples/pec_scattering.py --shape slot --ppw 24 --bins 9e8 1e9 1.1e9 --output artifacts/slot
.venv/Scripts/python.exe examples/pec_scattering.py --shape pair --ppw 32 --graded --output artifacts/pair
```

The example uses a six-wavelength square and a `6*ppw` square cell budget. The
`--graded` option selects the deterministic geometry-density strategy; it is a
demonstration allocation, not a tuned geometry baseline or a learned model.

## Notebooks and API guide

- [Geometry, mesh and enlarged cells](notebooks/01_geometry_and_mesh.ipynb): CPU
  preparation, exact geometry, uniform/deterministic meshes, source spectrum, and
  CNN raster inputs.
- [GPU scattering and far field](notebooks/02_gpu_scattering_and_far_field.ipynb):
  one 21-bin solve, per-bin convergence, scattering width, complex far field,
  analytical-cylinder comparison, and HDF5 reload.
- [Solver/API reference](docs/solver_api.md): settings, material overlays, mesh
  strategies, checkpoint format, archives, and plotting options.
- [Geometry-aware optimization study](docs/geometry_aware_optimization_study.md):
  strict-conformal PEC meshing, engineered-shape angle screens, fixed-budget
  GPU comparisons, numerical-reference checks, and a constrained CNN proposal.

Install the `notebooks` dependency group and select the project `.venv` kernel.
Both notebooks include executed example outputs.

## Geometry

`Simulation` owns one continuous `Geometry`. Add primitives in the order in which
they should be applied:

```python
outer = sim.add_rectangle(
    (2.5 * lam, 3.5 * lam), (2.55 * lam, 3.45 * lam), material="PEC"
)
sim.add_rectangle(
    (2.87 * lam, 3.13 * lam), (2.85 * lam, 3.5 * lam), material="air"
)
sim.add_polygon(
    [(2.0 * lam, 2.0 * lam), (2.3 * lam, 2.1 * lam), (2.2 * lam, 2.5 * lam)],
    material="PEC",
)
sim.remove_geometry(outer)
```

`add_circle(center, radius, material="PEC")`, `add_rectangle(x, y,
material="PEC")`, and `add_polygon(vertices, material="PEC")` accept only `PEC`
or `air`. Every primitive must lie strictly inside the domain. Later primitives
overlay earlier interiors, so an air primitive can carve a vacuum hole in a PEC
primitive. Removing or adding geometry invalidates the applied mesh, prepared
coefficients, and previous `sim.result`; apply a mesh again before solving.

Continuous geometry is the source of truth. Display rasterization does not define
the solver geometry. Analytic circles and simple non-self-intersecting polygons are
supported. Unresolved cut topology, missing enlarged-cell donors, disconnected open
segments, and hidden subcell gaps are rejected. Refinement does not guarantee that
every corner alignment is supported; adjust geometry, anchors, or the budget.

## Mesh strategies

```python
sim.apply_mesh("uniform", cells=(144, 144))
sim.apply_mesh("deterministic", cells=(144, 144))
sim.apply_mesh("density", cells=(144, 144), density=(rho_x, rho_y))
sim.apply_mesh("cnn", cells=(144, 144), checkpoint="models/pec_mesher")
sim.apply_mesh(existing_mesh)
```

The named strategies require an exact `(Nx, Ny)` budget, including fixed PML
collars and compulsory layout anchors. `uniform` targets constant spacing but may
need to adjust lines for those hard coordinates. Pass `strict=True` to reject an
incompatible exact-uniform request instead of allowing the uniform preference to be
projected through the constrained density mesher. `strict` is valid only for the
uniform strategy. An explicit `Mesh` is used as supplied and cannot be combined
with generator options.

`deterministic` rasterizes resolved PEC occupancy, builds smoothed edge activity,
and projects the resulting axis densities. It is a reproducible geometry rule,
not an accuracy guarantee. `density` accepts two positive finite one-dimensional
vectors over equal-width bins. All generated meshes retain the exact budget,
anchors, PML lines, and adjacent-width grading constraints. Infeasible requests
raise `MeshInfeasibleError`; an unfinished optimizer raises `MeshOptimizationError`.

The default PML has 12 collar cells per side and physical thickness one half of the
layout wavelength on each enabled axis. Uniform collar placement is a hard
constraint. Geometry and the TFSF/contour layout must remain inside the permitted
interior guards.

## CNN strategy contract

CNN inference is optional and lazy. Install it with:

```powershell
uv sync --extra cnn
```

No trained checkpoint is supplied. A checkpoint is a directory containing
`model.onnx` and `manifest.json`. The manifest must declare:

```text
schema:        fdtdmesh-cnn-v1
method:        tmz-conformal-ect-v1
normalization: unit-domain
outputs:       positive-axis-density
shape:         raster height/width
channels:      geometry raster channels
sha256:        SHA-256 of model.onnx
```

The ONNX model must expose `rho_x` and `rho_y` outputs shaped `(1, bins)`. The
runtime supplies the manifest's raster channels plus four physical features:
domain-x/wavelength, domain-y/wavelength, `fmin/centre_frequency`, and
`fmax/centre_frequency`. The model supplies preferences only; the deterministic
mesher still owns counts, collars, anchors, spacing, and grading.

## Results, plots, and archives

`Result` is an immutable, plot-ready snapshot. It exposes `frequencies`, `angles`,
complex `far_field`, `scattering_width`, `history`, `bin_history`, `geometry`,
`mesh`, `diagnostics`, `configuration`, `converged`, and optional `debug` arrays.
The normal solve downloads final far-field amplitudes, widths, and diagnostics;
`diagnostic_download=True` is an explicit validation path for final fields/currents.

Both `Simulation` and `Result` support HDF5 save/load:

```python
sim.save("artifacts/cylinder/simulation.h5")
sim2 = Simulation.load("artifacts/cylinder/simulation.h5")

result.save("artifacts/cylinder/result.h5")
loaded = fdtdmesh.Result.load("artifacts/cylinder/result.h5")
```

`Simulation.save` stores the definition and any applied mesh. `Result.save` stores
the geometry, mesh, configuration, far-field arrays, scattering widths, convergence
history, and diagnostics. A loaded result can be inspected and plotted without a
CUDA runtime.

The following methods return a Matplotlib `Figure`; each accepts an optional `ax`
where supported by the method:

```python
sim.plot_geometry(mesh=True)
sim.plot_mesh()
sim.plot_discretization()
sim.plot_source()

loaded.plot_geometry(mesh=True)
loaded.plot_mesh()
loaded.plot_convergence()
loaded.plot_scattering(normalize="wavelength")
loaded.plot_far_field(component="phase")
```

`result.save_plots(directory)` writes the standard geometry, mesh, convergence,
scattering, amplitude, and phase PNGs. Matplotlib is included in the notebook
dependency group:

```powershell
uv sync --group notebooks
```

## Numerical conventions and limits

The phasor convention is `exp(+iωt)` and outgoing radiation uses Hankel functions
of the second kind. With `k=2πf/C0`, the stored two-dimensional width is
`4*abs(S)**2/k`, where `S` is the dimensionless complex far-field amplitude
normalized to the incident DFT at the phase origin. Width is in metres. Normalize
by `C0/f` for `σ₂D/λ`; a dB display uses `10*log10(σ₂D/λ)` with a plotting floor.
Complex phases from different origins are not directly comparable.

Automatic GPU stopping checks current and incident DFT changes for every requested
bin and the maximum residual field after a source/propagation guard. This certifies
the configured temporal settling policy; it does not certify spatial convergence,
PML independence, or accuracy for a high-Q or unusually sensitive geometry.
Compare meshes with the same physical geometry, analysis band, layout, and stopping
policy while independently refining cells and checking contour/PML sensitivity.

The public `Simulation`/`Result` API is the compatibility layer for new user code.
Lower-level `ScatteringCase`, `run_scattering`, `prepare`, native runtime, and mesh
projection functions remain available for tests and specialized numerical work, but
their signatures and internal layout details are less stable.

## Numerical mesh optimization

The [optimization API reference](docs/mesh_optimization_api.md) lists all public
study functions and their arguments in tables.

[Notebook 03](notebooks/03_reference_and_optimized_mesh.ipynb) builds qualified
fine-grid references and searches for meshes at a fixed cell budget using
`strategy="differential_evolution"` or `"powell"`. It includes 12 exact procedural
shapes and an optional multi-incidence sweep. See the [study API guide](docs/mesh_optimization.md)
for qualification, resume, server commands, and current limitations.

The notebook examples now fit the domain with
`sim.fit_domain(scatterer_margin_cells=5, exterior_cells=(6, 4))` before meshing.
This reserves, per side, 5 cells from final PEC bounds to TFSF, 4 to the contour,
and 6 to PML. With 12 PML cells and a 192-cell axis, 138 cells remain across the
scatterer bounding box. `make_simulation` uses this compact layout by default.

Geometry-first setup is available with `Simulation(fmin=..., fmax=...)` and
`apply_mesh("geometry_aware")`. See the [automatic domain guide](docs/geometry_aware_meshing.md)
and [Notebook 04](notebooks/04_geometry_aware_meshing.ipynb) for fixed exterior
allocation, exact input-coordinate geometry, and deterministic topology/donor repair.
