# Fixed-budget optimization of strict-conformal PEC meshes

## Finding

There is useful optimization headroom after a valid geometry-aware mesh has been
constructed. In this study, a 60-evaluation density search reduced the complex
far-field error beyond the measured fine-reference discrepancy for a rotated
star, a Wi-Fi symbol, a larger sun, and three engineered silhouettes. The gain
is strongly geometry dependent. The same search gave no resolved gain for two
circles, a rectangle, a moon, a small sun, or a U-shape. The small sun and U-shape
also rejected nearly all proposals, so their result says more about the search
parameterization than about the best possible mesh.

The geometry-aware constructor produced a strictly usable mesh in 44 of 48
catalogue shape/orientation/scale cases. Among those 44 valid geometry-aware
cases, a generic uniform grid at the same cell counts was usable in only 34.
This is a
feasibility result, separate from the accuracy gains below.

![Error reduction compared with measured reference discrepancy](figures/geometry_optimization/gain_vs_reference_discrepancy.png)

## Numerical setup

The model is **2D TMz scattering from PEC in air**, with a +x incident plane
wave, a Gaussian broadband pulse from 0.9 to 1.1 GHz, 21 DFT frequencies, and
360 observation angles. Lengths in the solver are SI metres. The source,
time-stepping, current DFT, convergence check and near-to-far-field calculation
run through the native CUDA solver in float64. The exact geometry remains a
continuous CSG recipe until meshing. A 30° incidence case is represented by
rotating that recipe relative to the fixed +x source; both reference and coarse
solutions use the same rotated geometry and angle coordinates.

The automatically generated domain uses 12 PML cells, six cells from the PML
interface to the integration contour, four from the contour to the TF/SF box,
and five from the scatterer to the TF/SF box. These policy lengths and all
exterior-axis coordinates stay fixed across coarse candidate meshes within a
case. This controls the boundary and source geometry when comparing meshes.
The fine-grid references intentionally refine their exterior as part of their
convergence study.

All candidates pass the solver's actual preparation step, including PEC/air
edge intersection and enlarged-cell donor checks. An invalid candidate is
recorded as a failed trial and never receives a far-field accuracy score.
The density search uses six controls per axis, differential evolution with
population 12 and seed 0, and at most 60 evaluations including the
geometry-aware, uniform, and deterministic baselines. Thus there are at most
57 search proposals per case. Every candidate has the same `(Nx, Ny)` as its
geometry-aware seed, preserves that seed's exact repair witness anchors, and
uses the same domain, frequency vector, material, TF/SF box, contour and PML.
The search changes only interior axis spacing. The stored per-trial archives
also record rejected conformal geometries and projection timeouts.

For each frequency, error is the angular complex-field relative L2 difference
from the qualified fine reference. The table reports the RMS of this error
over the 21 frequencies. Complex fields retain phase; this is stricter than a
comparison of scattering-width magnitudes alone. A tighter device stopping
run of each selected best mesh passed the reference tolerance.

The ordinary reference protocol uses uniform-target, exact-corner-aligned
grids. Qualification requires two successive spatial differences below 0.5%
in both RMS and worst-frequency metrics, plus separate checks with tenfold
tighter stopping thresholds, expanded PML and a displaced near-field contour.
The telescope's 33-segment dish prevents that exact-anchor projector from
producing a fine grid. Its 0° reference instead subdivides every interval of
its already-valid geometry-aware grid by factors 2, 3, 4 and 5, while refining
the PML cell count. Factors 3→4 and 4→5 pass the same spatial tolerance;
the same temporal, PML and contour checks pass. This preserves strict
conformal validity and gives a qualified **numerical** reference, though its
refinement family differs from the other cases.

The “reference discrepancy” below is the largest observed RMS difference
among the last two spatial comparisons and the temporal/PML/contour checks.
It is an empirical resolution indicator, **not a rigorous error bound**. A
gain smaller than this difference is unresolved by the present study. A gain
larger than it is evidence for useful headroom, not proof of global optimality
or of the exact continuum error.

## Equal-budget GPU results

Catalogue `scale` is in **metres**. Engineered `scale` multiplies the free-space
wavelength at 1 GHz. Errors and discrepancies are percent; gain is the
percentage-point reduction relative to the geometry-aware seed. “Valid” counts
only the 57 search proposals, excluding the three baselines. All 12 references
were qualified and all 12 selected meshes passed tighter stopping validation.

| Shape, angle, scale | Cells | Valid | Seed error | Best error | Gain | Reference discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| Circle, 0°, 0.2 m | 85×85 | 57/57 | 0.685% | 0.619% | 0.067 pp | 0.137 pp |
| Rectangle, 0°, 0.2 m | 90×78 | 57/57 | 1.161% | 1.161% | 0 | 0.336 pp |
| Star, 30°, 0.2 m | 94×94 | 39/57 | 3.202% | 2.374% | 0.828 pp | 0.328 pp |
| Moon, 30°, 0.2 m | 86×94 | 35/57 | 0.758% | 0.687% | 0.071 pp | 0.169 pp |
| Wi-Fi, 30°, 0.2 m | 93×84 | 12/57 | 1.961% | 1.721% | 0.240 pp | 0.057 pp |
| Sun, 0°, 0.2 m | 97×97 | 2/57 | 5.779% | 5.779% | 0 | 0.385 pp |
| U-shape, 30°, 0.2 m | 111×111 | 3/57 | 1.003% | 1.003% | 0 | 0.335 pp |
| Circle, 0°, 0.4 m | 116×116 | 57/57 | 0.787% | 0.735% | 0.052 pp | 0.245 pp |
| Sun, 0°, 0.4 m | 232×232 | 4/57 | 2.933% | 1.504% | 1.429 pp | 0.162 pp |
| Swept aircraft, 0°, 1λ | 178×159 | 17/57 | 2.230% | 1.760% | 0.470 pp | 0.053 pp |
| Propeller aeroplane, 0°, 1λ | 139×136 | 47/57 | 2.151% | 1.219% | 0.932 pp | 0.042 pp |
| Radio telescope, 0°, 1λ | 128×116 | 57/57 | 1.428% | 1.018% | 0.409 pp | 0.186 pp |

The deterministic baseline, scored at the same cell budget, is already better
than the geometry-aware seed for the moon, Wi-Fi and telescope. The winning
search mesh still beats that stronger baseline for Wi-Fi by 0.102 pp and for
the telescope by 0.211 pp. The telescope advantage over deterministic is only
slightly above its measured 0.186 pp reference discrepancy; it merits an
additional independent fine reference before using it as a training label.
For the moon, the gain over deterministic is below the discrepancy.

![Exact PEC shapes and their seed and selected Yee meshes](figures/geometry_optimization/selected_meshes.png)

The 1 GHz polar panels below show the *magnitude of complex far-field error*
versus observation angle, normalized by the reference field's angular RMS
magnitude. They show where each engineered mesh gained or lost angular
accuracy; the aggregate table is the broadband objective and does not imply
every angle improved.

![Angular complex far-field error for engineered silhouettes](figures/geometry_optimization/engineered_far_field_error.png)

## Engineered shapes and feasibility limits

The swept aircraft is a swept-wing/nacelle/tail top-view planform; the
propeller aeroplane is a straight-wing top-view planform with a broad PEC
propeller; the radio telescope is a 33-segment parabolic-dish side view with
a PEC mount and feed. The default feed support is electrically transparent;
explicit thin PEC struts proved much harder to mesh under this policy. These
are exact **2D polygon/CSG silhouettes** passed to the solver, then implicitly
extruded along z by TMz. They are useful geometry stress cases, **not 3D
aircraft or telescope radar cross-section predictions**. The dish polygon
approximates a parabola rather than representing an analytic curved surface.

Of 12 engineered combinations (three silhouettes × 0°/30° × 0.5λ/1λ),
seven produced valid strict-conformal geometry-aware grids within the
512×512/20-pass construction caps. All three 0°/1λ cases above ran the full
GPU reference and optimization study. The swept aircraft and propeller
aeroplane at 30° exhausted repair passes or the cell cap. The 30°/1λ
telescope did produce a valid 173×176 coarse grid, but its standard
exact-anchor reference had no valid fine level. A second, geometry-aware
fine-grid attempt converged at 40, 64 and 96 target points per wavelength;
its 40→64 change was 0.569%, above the 0.5% criterion, while 64→96 was
0.200%. Attempts at 112 and 128 target points per wavelength reached the
1024-axis-cell cap. There is therefore **no qualified 30° telescope accuracy
comparison**. We retain that result as an explicit reference-generation
limitation rather than treating the 96-grid field as ground truth.

For a further topology stress test, perforated PEC plates with 3×3, 5×5,
7×7 and 9×9 circular air holes were screened at 0° and 22.5°. Six of eight
obtained valid strict-conformal meshes; the 7×7 and 9×9 rotated versions hit
the 45-second projection cap. The 0° 9×9 case has 82 geometry primitives
and a valid 155×155 grid. These are feasibility screens only, not far-field
accuracy claims. A construction cap or timeout does not establish that no
valid mesh exists.

## Implication for a future learned mesher

The data support continuing mesh optimization for strict-conformal PEC
scattering. Large gains on the star, larger sun and two aircraft silhouettes
show that geometry awareness alone does not exhaust the fixed-budget accuracy
potential. The telescope result is promising but less decisive relative to its
strongest simple baseline. Feasibility is the immediate bottleneck for sharp
tips and dense CSG: only 2/57 small-sun proposals, 3/57 U-shape proposals,
4/57 large-sun proposals and 17/57 swept-aircraft proposals survived actual
conformal preparation. A CNN trained on arbitrary density vectors would waste
many outputs in the same invalid region. A next optimizer should encode
intersection/donor constraints directly or propose local changes around a
validated grid, then keep the strict solver validator as the final authority.
The failed direct-coordinate warp screen is recorded separately in the raw
study artifacts; all 168 sampled warps violated the present adjacent-ratio
constraint. That rules out that naive parameterization, not local mesh
optimization in general.

The committed [machine-readable results](geometry_aware_optimization_results.json)
contain every feasibility screen, reference level/check and aggregate search
result. The scripts in [`examples/geometry_optimization_study`](../examples/geometry_optimization_study)
recreate the screens, GPU references, 60-trial comparisons and figures. From
the repository root, for example:

```powershell
.venv/Scripts/python examples/geometry_optimization_study/probe.py
.venv/Scripts/python examples/geometry_optimization_study/engineered_probe.py
.venv/Scripts/python examples/geometry_optimization_study/perforated_probe.py
.venv/Scripts/python examples/geometry_optimization_study/reference_case.py swept_aircraft 0 1
.venv/Scripts/python examples/geometry_optimization_study/subdivision_reference.py 0 1
.venv/Scripts/python examples/geometry_optimization_study/optimize_case.py swept_aircraft 0 1 60
.venv/Scripts/python examples/geometry_optimization_study/build_report.py
```

Run `reference_case.py` and `optimize_case.py` with each catalogue/engineered
case in the table; use `subdivision_reference.py` for the 0° telescope.
HDF5 results and full per-trial traces are cached in `artifacts/`, keyed by
the complete simulation and optimizer descriptions. The committed JSON and
figures are the compact evidence record; the HDF5 caches remain local.
