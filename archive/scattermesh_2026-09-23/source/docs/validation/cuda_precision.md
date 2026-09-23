# CUDA float32 qualification — 22 September 2026

Float32 was tested against accepted float64 complex far fields for the settled
lossless epsilon_r=12 cylinder, the conductive epsilon_r=30 cylinder, and the
160² contour-qualified enlarged-PEC cylinder. The comparisons retain absolute
phase and use the same grids, source, duration, PML, monitor contour, and material
sampling as their float64 references.

The provisional precision gates are complex angular L2 difference below 0.1%,
weighted phase RMS below 0.1 degrees, analytic-reference error degradation below
0.1 percentage point, finite fields, and tail/peak below `1e-5`.

| Case | Float32/float64 complex difference | Phase RMS | Analytic-error degradation | Tail/peak | Decision |
|---|---:|---:|---:|---:|---|
| Lossless epsilon_r=12, 512²/400 ns | 0.00112% | 0.000437° | 0.000765 percentage point | 7.29e-6 | Pass |
| Conductive epsilon_r=30, 512²/50 ns | 0.00109% | 0.000502° | 0.000080 percentage point | 1.89e-7 | Pass |
| Enlarged PEC, 160²/35 ns | 0.000256% | 0.000143° | 0.000192 percentage point | 3.69e-7 | Pass |

Float32 passes this bounded accuracy qualification. It is not selected as the
production default because it provides no measured throughput gain on this TITAN
RTX/PyTorch implementation:

| 512², 10 ns | Float64 CUDA | Float32 CUDA | Float32 speed relative to float64 |
|---|---:|---:|---:|
| Dielectric | 4.791 s | 5.500 s | 0.87x |
| Enlarged PEC | 7.027 s | 8.964 s | 0.78x |

These short throughput probes include CUDA setup and choose the current execution
precision; they do not characterize every GPU architecture. Production references
remain float64. Float32 remains available for future batched/fused kernels or
hardware where a new benchmark shows an actual benefit.

Run individual precision cases and regenerate the summary with:

```bash
.venv/bin/python scripts/qualify_cuda_precision.py --case eps12_small --device cuda:0
.venv/bin/python scripts/qualify_cuda_precision.py --case eps30_lossy --device cuda:1
.venv/bin/python scripts/qualify_cuda_precision.py --case pec_enlarged --device cuda:2
.venv/bin/python scripts/summarize_cuda_precision.py
```

Local records and native complex arrays are under `runs/cuda_precision/`. The
summary also links the float32 and float64 benchmark reports used in the decision.
