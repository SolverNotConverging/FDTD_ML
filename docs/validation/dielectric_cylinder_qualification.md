# Dielectric-cylinder qualification — 22 September 2026

The restartable `scripts/qualify_dielectric_cylinders.py` campaign completed 25
selected cases. Six analytic cylinders cover epsilon_r 2, 4, 12, and 30, including
electric conductivities 0.05 and 0.2 S/m. All cases use mu_r=1 and sigma_h=0.
Base grids cover 64², 96², and 128². The initial epsilon_r<=4 family also uses
192², with independent duration, PML, near-to-far contour, and material-quadrature
variations for the lossless epsilon_r=4 cylinder.

The campaign writes atomic per-case JSON/NPZ records and validates the complete
physics-source/configuration fingerprint plus complex-array structure before resume.
Targets are analytic complex far fields at 0.8, 1.0, and 1.2 GHz over 180 angles.
Each record includes the shortest internal wavelength in cells so material contrast
and geometric feature resolution remain explicit model inputs and diagnostics.

## Initial-stage result

| Check | Measured | Gate | Result |
|---|---:|---:|---|
| Worst 192² complex angular L2 | 1.403% | <2% | pass |
| Worst 192² weighted phase RMS | 0.671° | <1.5° | pass |
| Worst 192² late-field ratio | 2.67e-8 | <1e-5 | pass |
| Analytic series-order change | 0 | <1e-10 | pass |
| Worst independent numerical variation | 0.232% | <0.5% | pass |

The accepted 192² cases have at least 19.99 cells per shortest internal wavelength.
The maximum complex-field changes from longer duration, thicker PML, a larger
near-to-far contour, and twice the material quadrature are 2.1e-9, 6.2e-7, 0.232%,
and 0.0167%, respectively.

## Deferred contrast stages

The epsilon_r=12 lossless cylinder has only 7.69 cells per shortest internal
wavelength at 128². It reaches 30.8% complex error, a 15.9° phase RMS, and a 0.0172
late-field ratio. Increasing material quadrature changes the complex field by up
to 1.42%, confirming that it is not reference-ready.

The epsilon_r=30 cases have only 4.87 cells per shortest internal wavelength at
128². The lossless case reaches 63.5% complex error and a 0.128 late-field ratio;
the conductive case reaches 15.5% error but settles much faster. These are retained
as high-contrast stress cases. They require finer spatial resolution based on the
internal wavelength and automatic time escalation before they enter accepted data.

Local artifacts are under `runs/dielectric_cylinder_qualification/`, including the
report, per-case arrays, and `qualification.png`. They are ignored by Git; this
document records the reproducible measured result.
