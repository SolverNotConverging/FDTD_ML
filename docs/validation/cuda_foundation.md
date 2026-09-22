# CUDA dielectric and PEC foundation — 22 September 2026

`scattermesh.simulate_cuda` ports the nonuniform TMz update, CPML, analytic
plane-wave source, and stagger-aware streaming surface DFT to PyTorch.
Material filling fractions are prepared once on the CPU. During time stepping,
fields, CPML state, source evaluation, and complex DFT accumulators remain on the
selected device. Only bounded scalar health checks and the final fields/DFTs return
to the host. DFT phases use recurrence with an exact analytic reanchor every 2,048
steps.

The backend supports dielectric, PEC-only, and separated mixed scenes with mu_r=1
and sigma_h=0. PEC circles/rectangles use staircase, conformal, or enlarged modes.
PEC geometry, cut intersections, and the enlargement operator are constructed once
on the CPU; inverse-length gradients, affine total-field constraints, and projected
mass transfers remain on the selected Torch device during stepping. Enlarged mixed
scenes require a vacuum transfer stencil; contacting/overlapping materials remain
rejected. See the [mixed qualification](mixed_scattering.md). Float32 is accurate in
the bounded [precision qualification](cuda_precision.md), but slower in both 512²
probes on this server, so production references remain float64.

## Equivalence

The real-GPU test on one NVIDIA TITAN RTX compares NumPy and CUDA float64 final
Ez/Hx/Hy fields, electric and magnetic surface DFTs, incident spectrum, and complex
normalized far field. All relative differences are below 5e-15 in the bounded
dielectric test. A second matrix compares all three PEC modes on Torch CPU and a
physical GPU, including final fields, electric/magnetic DFTs, incident spectrum,
complex far field, CFL fraction, and enlargement diagnostics. All nine backend
tests pass on the NVIDIA device. At 512² enlarged PEC, complex far-field difference
is 4.13e-15; final-field differences are 1.24e-15 (Ez), 2.10e-15 (Hx), and 1.66e-15
(Hy).

## Throughput

`scripts/benchmark_cuda.py` uses a 10 ns source-active run to measure update and
DFT throughput. It is not a settled-scattering accuracy qualification.

| Uniform grid | Nt | NumPy float64 | CUDA float64 | Speedup | Complex far-field difference |
|---|---:|---:|---:|---:|---:|
| 128² | 503 | 0.443 s | 3.333 s | 0.13x | 1.27e-15 |
| 512² | 2,010 | 34.646 s | 4.791 s | 7.23x | 4.36e-15 |

The matching enlarged-PEC benchmark uses the same displaced cylinder and reports:

| Uniform grid | Nt | NumPy float64 | CUDA float64 | Speedup | Complex far-field difference |
|---|---:|---:|---:|---:|---:|
| 128² | 503 | 0.719 s | 3.584 s | 0.20x | 1.20e-15 |
| 512² | 2,010 | 37.033 s | 7.027 s | 5.27x | 4.13e-15 |

Kernel-launch and CUDA-context overhead make the first small case slower. The fine
reference regime crosses decisively in favor of CUDA. The 512² final-field relative
differences are 1.31e-15 (Ez), 1.33e-15 (Hx), and 1.81e-15 (Hy).

The measured reports and plots are local under `runs/cuda_foundation/` and
`runs/cuda_pec/`. Each report
records the exact package hashes, environment, device properties, diagnostics, and
component errors. These artifacts are ignored by Git; this document preserves the
reproducible result.
