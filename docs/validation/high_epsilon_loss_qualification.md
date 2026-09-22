# High-permittivity lossy-cylinder qualification

The 70 ns qualification campaign tested whether high dielectric contrast can remain
in the simple-scene curriculum when material loss is controlled. Dataset
`high_epsilon_loss_91a11296bb4504fb` contains 18 analytic cylinders spanning
epsilon_r={10,20,30}, radii={0.05,0.12} m, and loss tangent={0.03,0.10,0.20} at
1 GHz. Each geometry was run at exact 32x32 and 96x96 budgets with uniform,
region-strong, and hybrid-wide axes, for 108 cases total. Conductivity was computed
as `sigma_e = tan_delta * 2*pi*f*epsilon_0*epsilon_r`, giving a tested range of
0.0167--0.3338 S/m.

Campaign `high_epsilon_loss_qualification_549a0f24bfb1e339` produced this settling
matrix:

| epsilon_r | tan delta 0.03 | tan delta 0.10 | tan delta 0.20 |
|---:|---:|---:|---:|
| 10 | 3/12 | 12/12 | 12/12 |
| 20 | 0/12 | 12/12 | 12/12 |
| 30 | 0/12 | 12/12 | 12/12 |

All 72 cases at loss tangent 0.10 or 0.20 passed the `tail/global < 1e-5`
criterion. Their largest accepted tail ratio was 3.77e-6. The low-loss 0.03 tier
is unsuitable for the fixed 70 ns campaign: 33/36 cases remained unsettled, often
with tail ratios between 1e-4 and 1e-3.

The high-loss result is also useful for mesh learning rather than merely easy to
settle. At epsilon_r=30 and 96x96, region-strong median joint scattering loss was
0.00769 for loss tangent 0.10 and 0.00132 for 0.20, versus 0.0428 and 0.00849 for
uniform. Across every settled geometry/budget group, a nonuniform policy won the
soft-Nt objective; the median selected improvement was 4.48x.

The simple curriculum therefore extends to epsilon_r=30. Above epsilon_r=10 it
uses loss tangent 0.10--0.20 in training, 0.14 in validation, and 0.12/0.18 in
test. Lower contrasts retain independent fixed-conductivity variation and the
lossless epsilon_r={2,4} controls. This condition avoids an arbitrary permittivity
cap while excluding the measured high-Q region that cannot settle within the
campaign duration.
