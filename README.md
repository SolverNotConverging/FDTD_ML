# PEC scattering and nonuniform meshing

A small research foundation for learning mesh allocation for **2D TMz PEC scattering**.
The current implementation covers restart stages 0–2: constrained tensor-product
meshes, continuous PEC geometry, conformal enlarged-cell FDTD, normal-incidence
TFSF, current DFTs, and bistatic near-to-far conversion. Training and mesh search
are the next stages.

**FDTD, current DFT accumulation, convergence decisions, and NF2FF run in native
CUDA.** The GPU stops itself using a conditional graph. Only final complex
far-field amplitudes, scattering widths, and small diagnostics return to the CPU.
Optional progress callbacks use asynchronous snapshots and do not control stepping.

The legacy dielectric/CNN experiments and original restart proposal are preserved
on `legacy-project` at `17b1fb7f83feb95ea102417a60cd58f8c6f8136e`. Development
continues on `master`. Old training code, generators, reports, and PyTorch
dependencies have been removed from the active project.

## Laptop setup

Requires Python 3.11+, an NVIDIA GPU, **CUDA Toolkit 13.x**, a compatible driver,
and a supported C++ compiler. Validated locally with Python 3.12, CUDA 13.3,
MSVC, and an RTX 4070 Laptop GPU (8 GB). There is no production CPU fallback.

```powershell
uv sync --group dev
./scripts/build_cuda.ps1
.venv/Scripts/python.exe -m pytest -q
```

The default native target is SM 89. Before building on another GPU, set
`FDTDMESH_CUDA_ARCHS` to its compute capability, e.g. `80,89`. The last entry also
gets PTX. Rebuild after changing the native source or architecture.

On a Linux server, the intended build is below; this platform has not yet been
qualified. Rerun the numerical checks there before using it for references.

```bash
uv sync --group dev
FDTDMESH_BUILD_CUDA=1 FDTDMESH_CUDA_ARCHS=80 uv run --no-sync python setup.py build_ext --inplace
uv run --no-sync pytest -q
```

## Run examples

```powershell
.venv/Scripts/python.exe examples/pec_scattering.py --shape cylinder --ppw 24
.venv/Scripts/python.exe examples/pec_scattering.py --shape slot --ppw 24 --bins 9e8 1e9 1.1e9 --output artifacts/slot
.venv/Scripts/python.exe examples/pec_scattering.py --shape pair --ppw 32 --graded --output artifacts/pair
```

Shapes: `empty`, `cylinder`, `rectangle`, `pair`, `slot`. `--ppw` is a nominal
budget, a multiple of eight. `--graded` uses the retained constrained density
projector; its optimization can take much longer than the GPU solve. It is a
demonstration density, not a tuned geometry baseline or a learned model.

Each example saves `scattering.npz`, `diagnostics.json`, and `scattering.png`.
NPZ contains frequencies, angles in radians, dimensionless complex amplitudes,
2D scattering widths in metres, convergence history, and physical x/y grid lines.
Outputs are ignored by Git under `artifacts/`.

No time window is required. `--check-interval` defaults to 2048 updates;
`--max-steps` is a failure cap. Nonconvergence raises an error and saves an
explicitly named `unconverged.npz`, rather than silently accepting a label.

```python
from fdtdmesh.cases import canonical_case
from fdtdmesh.simulation import run_scattering

case, mesh = canonical_case("cylinder", ppw=32)
result = run_scattering(case, mesh)
result.save("artifacts/cylinder.npz")
```

## Validation and scope

```powershell
.venv/Scripts/python.exe benchmarks/validate_scattering.py
```

The recorded cylinder width errors at 64 nominal cells per wavelength are
**0.137% uniform** and **0.077% graded**, against the independent cylinder series.
Both precision modes have CPU/GPU regression checks; Compute Sanitizer memcheck
reported zero errors. See [measured validation](docs/validation.md),
[numerical method and restrictions](docs/numerical_method.md), and
[remaining research stages](docs/pec_restart_plan.md).

Supported geometry consists of analytic circles and simple polygons with ordered
vacuum overlays. Unsupported cut topology or missing enlarged-cell donors raises
`UnresolvedGeometryError`. Arbitrary silhouettes may need geometry anchors or
refinement; refinement alone does not guarantee every corner alignment is valid.
The current solver is +x incidence, vacuum/PEC, TMz, and z-invariant. Its output
is 2D scattering width, not 3D RCS. Small demonstration cases are qualified;
high-Q resonators and general training references still need individual convergence
and boundary-sensitivity checks.

The NumPy implementation and CPU NF2FF exist only as test oracles. Explicit
`diagnostic_download=True` retrieves fields/currents **after** GPU NF2FF for validation;
normal runs do not download them.
