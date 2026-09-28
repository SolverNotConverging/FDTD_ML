# Geometry-first repository cleanup and hybrid boundary treatment

Status: implemented as the 1.0 breaking cleanup. This document records the
approved design; the API guides describe the shipped signatures. The project has
one user, so obsolete public call signatures and compatibility aliases were removed.
The current pushed commit `cce9c6e` is the reproducibility baseline. Existing
completed studies remain evidence for that version, not validation of the new
hybrid method. No new large simulation campaign is part of this cleanup.

## 1. One supported workflow

Define simulation physics -> add exact geometry -> apply mesh (resolve automatic
domain and prepare coefficients) -> inspect -> solve -> save/plot.

- Keep exact ordered PEC/air geometry independent of the domain and Yee grid.
  Later additions replace earlier material; touching PEC regions form a union.
  Domain bounds come from final material after CSG, not primitive/image canvases.
- Remove the public `size`, manual `layout`, manual `pml`, and `fit_domain()`
  construction routes. Retain an internal resolved-layout type for solver work,
  boundary sensitivity tests, and HDF5 restoration.
- Preserve input coordinates. Record any solver translation/rotation separately;
  default geometry plots and shape labels use the input frame.
- Resolve the domain only after geometry exists. Geometry edits invalidate layout,
  mesh, coefficients, and results. Failed preparation is transactional.
- Keep SI metres and Hz. Name all public angle arguments with their unit and
  distinguish observation angles from illumination and geometry orientation.
  The current +x source and rotation-based studies must be described honestly;
  a rotation wrapper must transform observation angles and phase origin too.
  Native arbitrary-angle injection is separate numerical work.

Automatic-domain distances are specified as fractions of the centre-frequency
wavelength, `lambda0 = c / f0`, with `f0 = (fmin + fmax) / 2`. They are independent
of the candidate interior cell budget. Proposed nominal defaults:

| Distance | Wavelength fraction |
|---|---:|
| Scatterer bounds to TFSF | `5/24` |
| TFSF to NF2FF contour | `4/24` |
| NF2FF contour to inner PML boundary | `6/24` |
| PML physical thickness | `12/24` |

These preserve the earlier ratios at 24 cells per centre wavelength, rather than
promising identical extents to the old `fmax`-based cell-count policy. Exterior
resolution is a separate setting: default maximum spacing `(c / fmax) / 24`, with
an optional explicit maximum spacing in metres. Keep a PML minimum resolution
of 12 cells independently of its requested physical thickness.

Exact arbitrary physical gaps and one identical spacing across every exterior
segment cannot always both be satisfied. Treat these fractions as requested
minimum clearances/thickness: choose one common exterior spacing satisfying the
resolution and PML requirements, and round each gap outward to whole cells. Report
both requested and achieved physical distances, cell counts, and spacing. The
rounding increment is less than one exterior cell per segment. Resolve once per
physical case, then freeze the resulting layout across candidate budgets and
strategies; never re-round it for each interior mesh. Reference qualification may
subdivide those cells at fixed boundaries or explicitly test larger clearances.
These defaults are practical starting settings, not universal boundary-convergence
guarantees. Tensor-product interior lines still change tangential sampling outside
TFSF.

## 2. Separate three independent choices

| Choice | Proposed API | Meaning |
|---|---|---|
| Axis generation | `apply_mesh(strategy=...)` | Where the Cartesian lines go |
| Boundary treatment | `BoundaryPolicy(mode=...)` | How PEC intersects the Yee operator |
| Mesh search | `optimize_mesh(strategy=...)` | How candidate grids are proposed and scored |

Boundary treatment must not be another axis-generation strategy: the same grid
must be usable with strict conformal or hybrid treatment for a controlled test.

Proposed call shape (illustrative, not executable with the current API):

```python
from fdtdmesh import Simulation, DomainPolicy, BoundaryPolicy, SolverSettings

sim = Simulation(
    fmin=0.9e9,
    fmax=1.1e9,
    domain=DomainPolicy(
        scatterer_to_tfsf_wavelengths=5/24,
        tfsf_to_contour_wavelengths=4/24,
        contour_to_pml_wavelengths=6/24,
        pml_thickness_wavelengths=0.5,
        exterior_ppw=24,  # Resolution uses fmax, distances use f0.
        min_pml_cells=12,
    ),
    boundary=BoundaryPolicy(mode="hybrid"),
    solver=SolverSettings(dft_bins=21),
)
sim.add_circle(center=(0.0, 0.0), radius=0.14, material="PEC")
sim.apply_mesh("geometry_aware", cells=(160, 160))
sim.plot_geometry(mesh=True, figsize=(16, 14))
sim.plot_discretization(show_fallback=True, figsize=(16, 14))
result = sim.solve(progress=True)
result.save("artifacts/circle/result.h5")
result.plot_scattering()  # Polar; 2D scattering width, not 3D RCS.
```

Keep `DomainPolicy` and `SolverSettings` rather than renaming everything. Replace
the overloaded uniform `strict` switch with explicit semantics:

| Grid strategy | Contract |
|---|---|
| `uniform` | Truly uniform axes, including collars; reject incompatible fixed layout/counts |
| `quasi_uniform` | Near-uniform interior projected through fixed layout constraints |
| `geometry_aware` | Exact-geometry feature allocation and validated boundary preparation |
| `density` | Supplied positive axis densities, followed by the common projector |
| `custom` | Explicit mesh supplied through a named `mesh=` argument |

Remove the raster heuristic named `deterministic` from the main API; keep its
implementation only if needed as an explicitly named research baseline. Remove
the untrained ONNX/CNN facade and optional dependency from the stable API for now;
retain density/custom proposal extension points for the future learned model.

Use two mutually exclusive resolution contracts: `cells=(Nx, Ny)` is an exact
total axis budget including PML; `target_spacing=...` with `max_cells=...` allows
automatic counts up to caps. Never silently exceed a fixed budget. Geometry-aware
fixed-budget seed construction is new work: try legal allocations and repairs
within that budget, report failure if unsuccessful. A construction timeout does
not prove mathematical infeasibility. Report reserved/free cells by axis before
expensive preparation. Put advanced projector/repair settings into one mesh-options
object with documented defaults rather than growing the method's keyword list.

## 3. Hybrid conformal/staircase policy

Ansys documents local staircase fallback when more than two materials occur in
a Yee cell. Its public documentation does not establish that this is identical
to our proposed fallback for every non-simple PEC/air cut. Call our implementation
`hybrid`, describe its own rules, and do not claim numerical equivalence to
Lumerical's proprietary conformal implementation.

Source: [Ansys mesh refinement options](https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object).

In this 2D TMz code, a cell has shared Ez nodes and staggered Hx/Hy edges; replacing
one scalar cell label is insufficient. Build the hybrid discretization as follows:

1. Classify final exact PEC/air geometry against cells and shared edges. Distinguish
   ordinary cuts, multiple edge crossings, enclosed subcell features, and missing
   enlarged-cell donors. A concave shape or a sharp corner alone is not a failure;
   classification is local. Different endpoint materials also do not guarantee
   a single crossing. Replace the current primitive-bounds shortcut with resolved
   CSG tests, including enclosed PEC islands and air holes.
2. For supported regions use the existing conformal coefficients and enlarged-cell
   updates unchanged. Small conformal faces still require valid enlargement.
3. For unsupported topology select a local staircase patch using material samples
   at the actual Ez locations and ordinary Yee coefficients. Exact input geometry
   stays intact; only its discrete representation changes. Thin PEC can disappear
   and narrow air gaps can close under sampling. Record these events explicitly.
4. Assign each shared degree of freedom and edge exactly once. Reconcile neighboring
   stencils and rebuild donor pairs after patch changes. Initially exclude fallback
   patches from donating to conformal pairs. Expand a patch only as required by
   shared-stencil consistency or an unresolved donor dependency, record the reason,
   and terminate on a finite mesh or a configured fallback limit.
5. Rebuild the operator and CFL bound for the actual mixed coefficients. Verify the
   symmetry/positive-semidefinite stiffness and positive mass properties used by
   the existing stability argument. An exception handler that ignores an invalid
   cut or replaces one length without checking its neighbors is not sufficient.

Two public modes:

- `conformal`: strict topology and enlarged-cell donor requirements; reject failure.
- `hybrid`: conformal wherever supported, local staircasing where unsupported.
  Donor-triggered fallback must be separately counted from topology-triggered
  fallback. Numerical failure must never be relabelled as successful fallback.

Keep strict conformal as the initial default and the numerical-reference policy.
Make hybrid an explicit option in the new examples; consider changing the default
only after its validation gates pass. A full-staircase internal test operator is
useful for limiting-case tests; it need not become a primary user workflow.

Preparation produces a structured report: raw unsupported cells/edges, final
fallback mask, triggering reasons, patch expansion, enlarged pairs, fallback area
fraction, boundary-intersecting-cell fraction, and detected unresolved features.
Warn once per preparation with a summary and plotting instructions. Counts alone
are not error bounds: one lost thin connection can matter more than many small
boundary displacements. Allow a maximum fallback fraction and `on_fallback="error"`
for callers requiring strict acceptance. Save diagnostics and exact geometry in HDF5.

## 4. Optimizer and reference API

- Promote reusable reference qualification and optimization from `benchmarks` into
  `fdtdmesh.optimization`; put shape recipes in `fdtdmesh.catalog`.
- Default search to `feasible_local`; retain DE as the comparison algorithm.
  Remove Powell unless a maintained example or scientific need justifies it.
- Expose search budgets in one settings object: proposal count, unique solve count,
  time limit, population, RNG seed, local radius/repair policy. No silent extra
  solves hidden behind the requested budget; separately label winner validation.
- Provide one qualified-reference service supporting ordinary refinement and nested
  seed subdivision, removing the duplicate telescope/reference scripts.
- Remove optimizer dependence on a particular mesh metadata producer. Consume a
  validated preparation/constraint object with fixed lines, witnesses, and policy.
- Strict searches preserve strict geometry constraints. Hybrid searches can explore
  topology changes through controlled fallback, but keep the fallback policy and
  limits fixed over the experiment. Report acceptance and accuracy separately.
- Preserve fixed exterior coordinates, exact axis counts, grading, and spacing.
  Keep the existing LP coordinate mobility estimates labelled as conditional upper
  bounds; add observed movement, acceptance and fallback diagnostics beside them.
- Compare either policy to the same qualified strict reference when possible.
  Hybrid fallback is mesh-dependent discretization error, so it must be included
  in the complex-field objective and separately reported. Do not imply that a
  smaller fallback area always means greater accuracy or monotone convergence.

## 5. Repository and documentation cleanup

| Current content | Planned treatment |
|---|---|
| `src/fdtdmesh/api.py`, `domain.py` | One automatic-domain public route; smaller typed settings |
| Public `Scene2D`, `ScatteringCase`, `run_scattering`, `Convergence` aliases | Remove top-level exports; keep needed internals private |
| `geometry_mesher.py` and `solver/conformal.py` | Share one topology classification/report, avoid divergent validity checks |
| `solver/reference_tmz.py`, canonical validation cases | Keep small CPU oracle/test fixtures; not a production CPU solver |
| `src/fdtdmesh/benchmarks/` | Split reusable optimization/reference code from geometry catalog and study orchestration |
| `examples/geometry_optimization_study/` | Consolidate repeated probes, reference runners and plotting into one configured study runner and one report builder |
| `examples/pec_scattering.py`, `examples/optimize_meshes.py` | Replace with short canonical solve and optimization examples |
| `docs/pec_restart_plan.md`, `docs/mesh_optimization_notebook_plan.md` | Remove completed planning documents from active docs; preserved in Git history |
| Existing study reports, JSON and figures | Preserve together under `reports/` with source commit and historical status |
| README and overlapping API/numerical guides | Rewrite around one quickstart, one API reference, one numerical guide, one validation guide |
| Old notebook implementations | Replace with the four maintained notebooks below |

New notebook set:

1. Geometry-first setup, CSG, automatic layout, mesh and exact Ez/Hx/Hy locations.
2. Native-CUDA solve, DFT-bin convergence, polar scattering/complex far field,
   analytical-cylinder check, HDF5 round trip.
3. Strict versus hybrid: thin gaps/strips, tips, enclosed features, fallback maps,
   identical-grid comparisons, and refinement error.
4. Qualified reference and fixed-budget optimization: several budgets and shape
   orientations, acceptance, variation, adaptivity, and large best-mesh plots.

Every notebook uses public imports, a consistent config cell, explicit output
directories, large mesh figures, and no duplicated solver/reference logic.
Expensive solves are opt-in. Do not ship stale execution outputs as new results.
Historical reports remain readable without the deleted experiment code through
their saved data/figures and pinned source revision.

Inventory notes: the two existing report builders cover different studies; freeze
each study's provenance independently rather than merging their scientific results.
The new consolidated runner/builder is for future studies. The older probe scripts
(`probe.py`, `engineered_probe.py`, `perforated_probe.py`) execute at import and should
be removed after required functionality is absorbed. New commands use a main guard.
No tracked HDF5 or build/cache purge is needed: `artifacts/` is already ignored.
Keep local expensive run archives; deleting those is not necessary for this cleanup.
Clear stale notebook outputs and machine-specific paths when migrating notebooks.

Version the archive schema and numerical method. Store exact design, coordinate
transform, resolved domain, mesh, boundary policy/masks, source, solver settings,
convergence and implementation identity. Cache reuse requires the complete relevant
description to match. Separate solver identity from optimizer orchestration and
plotting identity without dropping physics-sensitive inputs. Do not auto-convert
old results into new-method training labels. A small read-only historical loader
may live with report tooling if needed; no old simulation-construction API shim.

## 6. Delivery order and acceptance gates

1. **Inventory and baseline:** record the current revision, tests and archived
   study provenance; create a file-level keep/move/delete manifest.
2. **Breaking API cleanup:** automatic domain only, canonical settings, geometry
   and frame invariants, archive schema. Existing strict-conformal outputs must
   remain unchanged within established numerical tolerances.
3. **Shared preparation and hybrid operator:** classifier, staircase patch rules,
   shared-edge/donor consistency, diagnostics, CFL and native-CUDA compatibility.
   No change to GPU residency of FDTD, DFT, stopping or NF2FF.
4. **Meshing and optimizer integration:** exact-budget seed construction, policy-
   aware validation, unchanged physical case/exterior, consistent failure statuses.
5. **Examples/docs migration and deletion:** only remove superseded helpers after
   their reusable behavior and numerical tests have a maintained home.
6. **Small validation and release:** CPU geometry/operator tests, focused CUDA
   regression/smoke cases, clean notebook execution for CPU cells and a bounded
   GPU example, lint, archive round-trip/cache tests and documentation links.

Required numerical tests: ordinary cut unchanged; two-air-endpoint PEC crossing;
two-PEC-endpoint air gap; multiple crossings with different endpoints; enclosed
island/hole; shared-edge patch junction; unavailable/colliding donors; translated
and rotated input frames; variable spacing; geometry-only CSG overlays. Verify
strict rejection versus diagnosed hybrid preparation, matrix invariants/CFL,
native-CUDA versus small CPU oracle, and cylinder/complex-shape refinement.
Predefine acceptable field error and fallback limits per validation case before
inspecting results. The existing strict optimizer acceptance report does not
qualify the hybrid numerical method. Large 200-evaluation campaigns stay stopped.


## Implementation verification

The full regression suite passed 124 tests with native CUDA enabled. All four
notebooks were validated and their CPU cells executed; notebook 02 also completed
its 21-bin GPU solve and HDF5 round trip. The canonical circle example measured
0.749% relative scattering-width error at its centre frequency. The bounded
notched-PEC hybrid check and its limitations are in [validation](validation.md).
No large optimization campaign was restarted. Historical experiment scripts are
available at `cce9c6e`; the current command-line examples use the new public API.
