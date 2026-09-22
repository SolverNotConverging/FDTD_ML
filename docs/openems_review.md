# openEMS plane-wave implementation review

Inspected 22 September 2026, upstream commit
`8085f3129d4b41835e6e96365cb75218d60ef029` (21 September 2026).
This is a design review, not an openEMS simulation or accuracy benchmark.

## What the implementation does

The [TFSF operator](https://github.com/thliebig/openEMS/blob/8085f3129d4b41835e6e96365cb75218d60ef029/FDTD/extensions/operator_ext_tfsf.cpp)
accepts a box, normalizes propagation direction, snaps the box to mesh lines,
and constructs incident E/H corrections on its faces. It uses the actual Yee
coordinates, local edge lengths, and operator coefficients to build delays and
amplitudes. E and H corrections use their respective staggered locations.

The [time-stepping extension](https://github.com/thliebig/openEMS/blob/8085f3129d4b41835e6e96365cb75218d60ef029/FDTD/extensions/engine_ext_tfsf.cpp)
applies these corrections after voltage/current updates and linearly interpolates
between delayed signal samples. This supports fractional time-step propagation
delays rather than rounding the wavefront to an integer time step.

The [phase-velocity calculation](https://github.com/thliebig/openEMS/blob/8085f3129d4b41835e6e96365cb75218d60ef029/FDTD/operator.cpp)
uses average cell widths across the box and an optional specified frequency.
It estimates a numerical wave number from a Cartesian dispersion relation,
using Newton iteration for oblique incidence. The TFSF setup falls back to the
background physical wave speed when correction is disabled or invalid.

The [official source documentation](https://docs.openems.de/en/latest/concepts/excitations.html)
requires the source box to surround the scatterer in a homogeneous background,
with no material crossing the injection boundary.

## Implications for this project

An existing-line TFSF rectangle is compatible with nonuniform meshes and does not
require the old point TX/RX anchor system. It still needs a suitable vacuum margin,
correctly located electric/magnetic corrections, and leakage qualification. The
average-spacing phase correction is not proof of broadband accuracy on arbitrary
strongly graded grids; that is an inference from its inputs and formulation,
not an observed openEMS failure.

Our initial CPU solver uses an independently implemented pure scattered-field
formulation: the analytic incident field drives dielectric contrast throughout
the scatterer. There is no TFSF box. This choice removes boundary-injection error
from the initial formulation but still leaves material, propagation, PML, and
far-field discretization error. Empty vacuum being exactly zero is a construction
property; analytic nonempty benchmarks are essential.

Next comparison: add TFSF behind an explicit source option, retain the same mesh
and observation conventions, and compare both formulations against cylinders,
then PEC benchmarks. For TFSF, measure empty-domain leakage separately. Decide
the production source after these tests. The plan does not assume that either
formulation is universally more accurate.

No upstream implementation files were incorporated into `scattermesh`; public
source was inspected as a numerical design reference. openEMS is GPL-licensed.
