# PEC scattering and nonuniform meshing

FDTDMesh is a research solver for two-dimensional, z-invariant TMz scattering
from continuous PEC geometry. It uses a tensor-product nonuniform Yee mesh,
conformal cut-face coefficients, enlarged cells where required, a +x TFSF
incident wave, GPU current DFT accumulation, and GPU NF2FF conversion. The
reported scattering width is two-dimensional and measured in metres.

The public workflow is:

```text
Simulation -> add exact geometry -> apply_mesh -> solve -> Result
```

Geometry and mesh preparation run on the CPU. Native CUDA performs field updates,
DFT accumulation, device-controlled stopping, and NF2FF. There is no trained CNN
mesher or CNN checkpoint in the current runtime.

## Install and build

Python 3.11+, an NVIDIA GPU, CUDA Toolkit 13.x, a compatible driver, and a C++
compiler are required. The usual development commands are:

```powershell
uv sync --group dev
./scripts/build_cuda.ps1
.venv/Scripts/python.exe -m pytest -q
```

## A first solve

All coordinates and distances are SI metres. `fmin` and `fmax` are hertz. The
automatic-domain form derives the computational box from final PEC bounds.

```python
from fdtdmesh import BoundaryPolicy, DomainPolicy, Simulation, SolverSettings

sim = Simulation(
    fmin=0.9e9, fmax=1.1e9,
    solver=SolverSettings(),
    domain=DomainPolicy(),
    boundary=BoundaryPolicy(),
    observation_angles_deg=(0, 30, 60, 90),
)
sim.add_circle((0.0, 0.0), 0.17, material="PEC")
sim.apply_mesh("geometry_aware")
result = sim.solve(progress=True)
result.save("artifacts/cylinder/scattering.h5")
```

Domain policy gaps are expressed in wavelengths at the band centre. Exterior
clearances are rounded outward to whole cells once. Exact geometry is translated internally
to the solver frame; it is never rasterized to define the physical object.

## Geometry and mesh strategies

`Simulation` owns one continuous `Geometry`. Add circles, rectangles, polygons,
or ellipses in metres. Later overlays can carve air from PEC; adding or removing
geometry invalidates the mesh and result.

```python
sim.apply_mesh("uniform", cells=(144, 144))
sim.apply_mesh("quasi_uniform", cells=(144, 144))
sim.apply_mesh("geometry_aware", cells=(144, 144))
sim.apply_mesh("geometry_aware", target_spacing=sim.wavelength / 32,
               max_cells=(512, 512))
sim.apply_mesh("density", cells=(144, 144), density=(rho_x, rho_y))
sim.apply_mesh("custom", mesh=existing_mesh)
```

`uniform` may reject a fixed-layout request that cannot remain exactly uniform.
`quasi_uniform` projects through mesh constraints. `geometry_aware` accepts an
exact cell budget or a target spacing with caps, and validates exact geometry,
donors, and enlarged-cell topology. `density` uses positive axis densities at
an exact budget. `custom` accepts only `mesh=`. Advanced settings belong in
`MeshOptions`; see the [solver API](docs/solver_api.md) and
[mesh strategy guide](docs/mesh_strategy.md).

## Solver and results

`SolverSettings` controls DFT bins, precision, CFL safety, and the `DFTConvergence`
policy. Device stopping remains independent of spatial, PML, and boundary
sensitivity checks. `Result` provides scattering width, complex far field,
convergence diagnostics, and HDF5 save/plot helpers.

## Notebooks and references

- `notebooks/01_geometry_and_mesh.ipynb` — exact geometry and CPU mesh setup.
- `notebooks/02_gpu_scattering_and_far_field.ipynb` — native CUDA scattering and NF2FF.
- `notebooks/03_hybrid_boundary.ipynb` — boundary policy and targeted checks.
- `notebooks/04_geometry_aware_meshing.ipynb` — qualified references and multiple-budget optimization.
- `notebooks/05_hybrid_tip_optimization.ipynb` — bounded tip repairs, graded interior margins, qualified hybrid references and mesh search.

The [optimization API](docs/mesh_optimization_api.md) documents
`optimize_mesh`, `SearchSettings`, `FeasibleSettings`, `ReferenceSettings`,
`qualify_reference`, and `analyze_mesh_adaptivity`. The default feasible-local
search uses hybrid boundary treatment and auto-generates a geometry-aware
seed when none is supplied. Strict search remains an explicit policy override. Reference
subdivision requires an initial mesh. A bounded hybrid screen and its limitations are recorded in
[validation](docs/validation.md).

The [hybrid tip experiment](docs/hybrid_tip_optimization.md) records notebook 05's
measured optimization improvements and the star's unresolved reference convergence.

Historical studies are preserved in [reports](reports/), with their source
revision recorded there. They are historical evidence and do not validate the
hybrid method.

## Small examples

```powershell
.venv/Scripts/python examples/solve.py
.venv/Scripts/python examples/solve.py --run
.venv/Scripts/python examples/run_study.py examples/study.json
.venv/Scripts/python examples/build_report.py artifacts/study
```

The study runner also requires `--run` for GPU reference/search work. All notebooks
ship without execution outputs; GPU cells are opt-in. Install the notebook group
with `uv sync --group notebooks --group dev` and use the project kernel.

Version 1.0 intentionally removes the old manual-domain API and benchmark-package
imports. New HDF5 files use `fdtdmesh-h5-v2`; old archives are not silently reused
as results for the new method. Saved historical reports remain available.
