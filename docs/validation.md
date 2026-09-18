> Historical stage-one validation. Current grading, CPML, source, and receiver
> contracts and fresh results are in [stage 2 validation](stage2_validation.md).

# Stage-one validation

Validated on 2026-09-18 on Windows with an NVIDIA GeForce RTX 4070 Laptop GPU
(8 GB), NVIDIA driver 596.49, CUDA toolkit 13.3, MSVC 14.51.36231,
Python 3.12.14, PyTorch 2.14.0+cu130, NumPy 2.5.3, and SciPy 1.18.1.
Dependencies are pinned in `uv.lock`; `uv sync --locked --python 3.12` was verified.
Final suite: **46 tests passed**, including real-CUDA tests (no skips), in 2.69 s.
Ruff lint and formatting checks passed.

## Numerical checks

The native CUDA solver agrees with the NumPy oracle for all three final fields
and receiver histories on uniform and nonuniform meshes. Those cases include
lossy dielectric and magnetic materials, curved PEC geometry, duplicate point
sources, and finite line receivers. Tolerances are `rtol=2e-5, atol=2e-6`
for float32 and `rtol=1e-11, atol=1e-13` for float64. Zero-field runs stay exactly
zero; an anchored PEC wall gives exactly zero transmission to the opposite side.

An independent analytical rectangular-cavity standing wave checks the curl signs,
staggering, dual spacing, boundary enforcement, and convergence. Domain:
20 mm × 15 mm. Error is final Ez L2 error divided by initial Ez L2 norm after
approximately 1.3 periods, evaluated at the actual final simulation time.
The nonuniform axes are smooth sinusoidal perturbations of uniform coordinates.

| Cells per axis | Uniform vacuum error | Nonuniform vacuum error | Nonuniform lossy magnetic error |
|---:|---:|---:|---:|
| 20 | 7.80697e-4 | 1.22931e-2 | 1.02996e-2 |
| 40 | 1.93953e-4 | 3.11934e-3 | 2.61811e-3 |
| 80 | 4.87666e-5 | 7.86537e-4 | 6.61146e-4 |

The lossy case has epsilon_r=4, mu_r=2, sigma_e=0.04 S/m and uses the analytical
damped-cavity solution, with H initialized at -dt/2. Halving mesh spacing reduces
error by about four, consistent with second-order convergence for these smooth
problems. These checks do not establish convergence order for staircase interfaces
or subpixel features.

## GPU residency and integration

All field/coefficient/source/receiver buffers are allocated/uploaded before the
native time loop and downloaded after it. Runtime counters report zero transfers
while stepping. For source-free runs without monitors, upload/download byte counts
are unchanged between 3 and 100 timesteps. Source and receiver memory grows with
Nt when enabled. This is runtime accounting plus source inspection, not a separate
Nsight/CUPTI transfer trace.

Checkpoint round-trip preserves model predictions. Tests verify finite positive
axis densities, axis orientation, FiLM/convolution gradient flow, exact line budgets,
mandatory PEC anchors, physical scaling of input conditioning and raster maps,
and incompatible checkpoint rejection. The end-to-end test loads a checkpoint,
predicts densities, constructs an anchored mesh, then runs CUDA and checks nonzero
receiver output and repeatability. No trained model quality is claimed.

## Small timing experiment

Command: `python benchmarks/benchmark_solver.py --nx 128 --ny 96 --repeats 3`.
Same 20 mm × 15 mm vacuum domain, point source/receiver, 1 ns requested physical
duration, float32. One warm-up run, then three measured runs; median times below.

| Grid | Nt | Cell updates | GPU time | Total wall time |
|---|---:|---:|---:|---:|
| Uniform | 2,857 | 35,106,816 | 69.1 ms | 70.4 ms |
| Nonuniform density bump | 4,406 | 54,140,928 | 100.6 ms | 102.1 ms |

The smaller minimum spacing raises the nonuniform grid's timestep count at the
same physical duration. These are laptop/WDDM measurements, not a general speed
guarantee or accuracy comparison. Native launch overhead remains relevant on
small grids; CUDA graph replay and batched simulations are possible later improvements.

## Reproduce

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
.\scripts\build_cuda.ps1
.venv\Scripts\python.exe -m pytest -q --basetemp=artifacts/pytest
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe examples\simple_uniform.py
.venv\Scripts\python.exe examples\simple_nonuniform.py
.venv\Scripts\python.exe examples\cnn_mesh.py
```

The supplied examples ran successfully, including CNN inference on CUDA and native
FDTD in the same process. The CNN example labels its randomly initialized model.
CPML, training, batching, and interface/subpixel convergence remain outside stage one.
