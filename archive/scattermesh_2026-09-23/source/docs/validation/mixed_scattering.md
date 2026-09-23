# Separated mixed PEC/dielectric qualification

The CPU and CUDA solvers now combine dielectric contrast updates with exact
staircase/conformal/enlarged PEC boundary gradients for separated objects. PEC and
dielectric primitives that overlap or touch are rejected because their interface
priority and cut-material operator are not yet defined.

For enlarged PEC, every slave/root transfer node must remain in vacuum. Ordinary
dielectric nodes outside that local transfer stencil use the full conductive update,
including the `(ca - 1) Ez` increment before projected PEC advancement. A dielectric-
loaded transfer stencil is rejected rather than applying the vacuum Galerkin mass
silently. Conformal mode remains available for separated close objects when that
local enlargement restriction is encountered.

Torch CPU and physical-CUDA tests compare NumPy final fields, electric and magnetic
surface DFTs, incident spectrum, and complex far fields for both conformal and
enlarged mixed scenes. The complete GPU backend matrix reports 13 passes. PEC total
field is checked during the active pulse; overlap and dielectric-loaded enlargement
stencils have explicit rejection tests.

## Complex-field convergence

Three mixed scenes were evaluated at 0.8, 1.0, and 1.2 GHz. Each accepted profile
requires finite fields, tail/peak below `1e-5`, total electric field zero on PEC
nodes, spatial change below 1%, and duration/quadrature/contour changes below 0.5%.
Changes are angular complex-field L2 norms with absolute phase retained.

| Scene and accepted base | Spatial probe | Spatial | Duration | Quadrature | Contour | Maximum tail |
|---|---|---:|---:|---:|---:|---:|
| Wide-gap PEC + eps_r=4 circles, 256²/40 ns | 320²/40 ns | 0.123% | 0.000039% | 0.0120% | 0.231% | 3.28e-7 |
| 15 mm-gap PEC + eps_r=8 circles, 320²/240 ns | 384²/240 ns | 0.433% | 0.000012% | 0.0707% | 0.161% | 5.15e-6 |
| PEC rectangle + lossy eps_r=12 circle, 256²/55 ns | 320²/55 ns | 0.392% | 0.000105% | 0.0214% | 0.321% | 9.52e-6 |

The close-gap 40 ns profile had tail/peak about 0.023 and failed spatial and duration
checks. At 160 ns its tail was still about `4.4e-5`; the gate was not weakened.
The final 240 ns base and 300 ns duration probe settled at `5.15e-6` and `1.48e-6`.
Likewise, the rectangle/lossy 40 ns base had tail/peak `2.82e-4`; moving the base to
55 ns resolved it.

These results establish bounded self-convergence and backend equivalence for the
declared separated scenes. They are not an analytic proof of absolute multi-object
accuracy. External solver comparison, contacting interfaces, dielectric-loaded
aggregation, more objects, and thin screens remain separate qualification work.

Reproduce a scene profile and aggregate the accepted reports with:

```bash
.venv/bin/python scripts/qualify_mixed_scattering.py \
  --scene close_gap_circles --profile close_final --device cuda:0
.venv/bin/python scripts/summarize_mixed_scattering.py
```

Local complex spectra, per-variant records, and the aggregate report are under
`runs/mixed_scattering_qualification/`.
