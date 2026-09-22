# CUDA dielectric foundation — 22 September 2026

`scattermesh.simulate_cuda` ports the nonuniform dielectric TMz update, CPML,
analytic plane-wave source, and stagger-aware streaming surface DFT to PyTorch.
Material filling fractions are prepared once on the CPU. During time stepping,
fields, CPML state, source evaluation, and complex DFT accumulators remain on the
selected device. Only bounded scalar health checks and the final fields/DFTs return
to the host. DFT phases use recurrence with an exact analytic reanchor every 2,048
steps.

The backend currently supports dielectric scenes with mu_r=1 and sigma_h=0. PEC
is rejected until cut edges and Galerkin aggregation have a separately tested CUDA
implementation. Float64 is the qualified starting precision; float32 remains an
explicit future accuracy study.

## Equivalence

The real-GPU test on one NVIDIA TITAN RTX compares NumPy and CUDA float64 final
Ez/Hx/Hy fields, electric and magnetic surface DFTs, incident spectrum, and complex
normalized far field. All relative differences are below 5e-15 in the bounded
test. The ordinary test run reports 28 passed and one CUDA-visibility skip; the
same CUDA test run outside the filesystem sandbox reports all three backend tests
passed.

## Throughput

`scripts/benchmark_cuda.py` uses a 10 ns source-active run to measure update and
DFT throughput. It is not a settled-scattering accuracy qualification.

| Uniform grid | Nt | NumPy float64 | CUDA float64 | Speedup | Complex far-field difference |
|---|---:|---:|---:|---:|---:|
| 128² | 503 | 0.443 s | 3.333 s | 0.13x | 1.27e-15 |
| 512² | 2,010 | 34.646 s | 4.791 s | 7.23x | 4.36e-15 |

Kernel-launch and CUDA-context overhead make the first small case slower. The fine
reference regime crosses decisively in favor of CUDA. The 512² final-field relative
differences are 1.31e-15 (Ez), 1.33e-15 (Hx), and 1.81e-15 (Hy).

The measured report and plot are local under `runs/cuda_foundation/`. The report
records the exact package hashes, environment, device properties, diagnostics, and
component errors. These artifacts are ignored by Git; this document preserves the
reproducible result.
