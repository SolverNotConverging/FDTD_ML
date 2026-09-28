# Validation of the geometry-first cleanup

The full native-CUDA regression run passed **124 tests**, with no skipped tests.
An additional public optimizer integration test was then added; the focused
11-test hybrid/API module also passed, bringing the verified total to 125 tests.

The regression suite retains the original strict-conformal physics fixtures and
adds wavelength-domain, fixed-budget, hybrid topology/operator, and HDF5 checks.
Run `.venv/Scripts/python -m pytest -q` after building the native extension.
CUDA tests skip explicitly if the compiled runtime or GPU is unavailable.

Hybrid tests cover narrow PEC/air features, enclosed islands and holes, multiple
crossings, CSG-overwritten primitives, unchanged supported cuts, fallback limits,
symmetric positive stiffness and its eigenvalue bound. A short native-CUDA test
compares fields and staggered DFTs with the independent NumPy stepping oracle.
The existing suite retains analytical-cylinder convergence, enlarged-cell energy,
GPU NF2FF/DFT comparisons and asynchronous telemetry/residency checks.

## Bounded hybrid accuracy screen

Reproduce with `python benchmarks/validate_hybrid.py`. This performs three small
native-CUDA solves, not a mesh-optimization campaign. The predeclared gate is a
relative RMS complex far-field difference below 10% from a finer strict run.

| Notched PEC body | Cells | Staircase cells | Complex difference from strict fine |
|---|---|---:|---:|
| Hybrid | 100 × 100 | 14 | 0.1587% |
| Hybrid | 120 × 120 | 10 | 0.04344% |
| Strict fine | 150 × 150 | 0 | — |

All three runs reached their DFT/field stopping criteria. Both hybrid runs passed
the screening gate. The fine run is **not a qualified reference**: this screen
has no independent boundary sensitivity or repeated spatial refinement checks.
It cannot establish an error bound, general monotone convergence, or accuracy on
other thin features. Saved measurements are in
[hybrid_validation_results.json](hybrid_validation_results.json).

The additional triangle screen produced no fallback; it is not evidence for
accuracy of staircase patches. Strict conformal remains the default, and reference
qualification always uses strict mode with spatial and stopping/PML/contour checks.

Earlier detailed qualification and optimizer campaigns are frozen in
[historical reports](../reports/README.md). Their measured results refer to their
recorded revision and must not be relabelled as hybrid validation.
