# API migration validation

Validated on 2026-09-27 with Python 3.12, CUDA 13.3, float64 fields, and the local
RTX 4070 Laptop GPU.

- Native CUDA extension rebuilt successfully.
- Full suite: **45 tests passed** (38 existing tests plus 7 API tests).
- API tests include geometry/mesh independence, continuous material overlays,
  strict mesh constraints, immutable numerical arrays, actual ONNX inference
  through a tiny untrained convolution fixture, Gaussian spectrum checks, HDF5
  round trips, GPU per-bin history, residency counters, and failure qualification.
- Existing numerical tests retain CPU-reference field/DFT comparisons, GPU NF2FF
  agreement, cylinder refinement, and slow-callback independence.
- Both notebooks executed from start to finish with stored outputs. Geometry,
  enlarged-cell, convergence, and analytical-comparison figures were inspected.
- CLI cylinder, slot, and deterministic-mesh pair examples completed and saved
  HDF5 archives and PNG figures.
- Ruff checks and formatting passed.

The new notebook's fixed-radius cylinder uses a 144×144 grid, 0.9–1.1 GHz,
21 simultaneous DFT bins, and 256-step convergence checkpoints. It stops at step
2304. Relative complex far-field L2 errors against the cylinder series are
1.17%, 1.62%, and 1.89% at 0.9, 1.0, and 1.1 GHz respectively. These measure
this example at this resolution, not the accuracy of arbitrary geometries or
the deterministic/CNN mesh strategies. The run reports zero stepping-field
transfers and zero diagnostic field/current downloads.

Compute Sanitizer was attempted with the virtual-environment launcher and the
direct interpreter, but repeatedly timed out before attaching. Its stalled
processes were stopped. **Memory-checker validation of this revision remains
unverified**; a prior revision's successful sanitizer log is not evidence for this
revision. Rerun on a host where sanitizer process attachment works, for example:

```powershell
compute-sanitizer --target-processes all --tool memcheck --error-exitcode 1 .venv/Scripts/python.exe -m pytest tests/test_api.py -m cuda -q
```

No trained CNN weights are provided or validated. The ONNX test validates the
checkpoint/inference/projection interface only.
