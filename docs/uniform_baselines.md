# Uniform and quasi-uniform baseline conventions

New evaluations use baseline policy `uniform_snapped_pec_pml_and_quasi_uniform_v2`.
Historical reports labelled the projected constant-density baseline "uniform";
those saved reports retain their original identities and must be interpreted
using the old convention.

| Strategy | Grid | PEC | Probes |
|---|---|---|---|
| `uniform` | Constant spacing on each axis; no PEC/probe anchors | Rectangle faces and wire fixed coordinates/endpoints move to nearest mesh lines | Original physical positions, existing bilinear receiver/current-source mapping |
| `quasi_uniform` | Constant density preference projected onto anchors, PML collars and grading | Original geometry, exact anchors | Exact anchors |
| `heuristic` / `cnn` | Predicted density projected onto the same constraints | Original geometry, exact anchors | Exact anchors |

PEC snapping modifies a temporary simulation only. It occurs before point
material assignment and sampled dielectric averaging, so both use the same
discretized PEC boundary. Equal-distance ties choose the lower coordinate.
Ordinary dielectric geometry is unchanged. If snapping collapses a rectangle
or a wire to zero extent, the candidate fails explicitly. A source whose
interpolation support overlaps PEC also fails under the existing source rule.
Legacy field-increment sources retain their existing nearest-node behavior;
the v6 corpus exclusively uses integrated point-current sources.

Uniform-grid PML cell counts round the requested thickness to the nearest whole
number of uniform cells. Half-cell ties choose the thinner collar, toward the
outer boundary. The physical interface can move by at most half a cell;
aligned budgets preserve the original thickness exactly. Requested/actual
thickness, cell count and interface displacement are recorded in `pml_snapping`
diagnostics. Domain size, Nx/Ny and constant spacing remain unchanged. Geometry
and probes must still remain strictly outside the absorber. Constrained meshes
retain their original physical thickness and prescribed PML cell counts.

Policy v1 rejected nonaligned PML interfaces, unintentionally excluding many
intermediate/rectangular budgets. The v2 campaign retries those pairs; unchanged
successful pairs can be imported only after checking identical uniform grid
coordinates, zero PML displacement, and matching model/teacher/reference inputs.

Reference generation retains its exact-geometry uniform grid and mandatory
anchors and constant physical PML thickness. It never invokes PEC or PML snapping.
Completed campaign artifacts remain
untouched. No averaging default changes with this baseline update; v6 physics
comparisons should explicitly enable `material_averaging="sampled"` for every
strategy, matching the reference campaign.

Physics search includes true uniform for error/work comparison and cost
normalization, but excludes it from legal anchored CNN target selection. The
quasi-uniform candidate and its mixtures remain eligible. Deduplication separates
snapped and exact-geometry candidates even when line arrays coincide. New run
identities include the baseline policy, preventing old results from being resumed
under the new meaning of uniform.

The in-flight teacher-preparation job uses its already loaded original code and
source snapshot. Its heuristic targets and optimizer settings are unaffected;
these baseline definitions apply to subsequent FDTD comparisons.
