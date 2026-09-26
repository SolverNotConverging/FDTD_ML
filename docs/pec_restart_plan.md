# Restart plan: scattering-aware nonuniform meshing for PEC objects

Prepared 2026-09-26 against commit `366508e`. This is a proposed implementation
plan, informed by the current source and the earlier *CNN Meshing for FDTD*
conversation. No solver replacement, data deletion, or training has been performed.

## Research question and first decision

Can a CNN allocate a **tensor-product nonuniform Yee grid** that achieves a given
far-field scattering accuracy with less total computation than a strong
geometry-based mesher?

For vacuum and PEC, an effective-index map provides little useful differentiation
in the propagation region. Geometry rules still provide a strong baseline. The
learning opportunity is whether electrical size, illumination, interacting objects,
slots, and cavities change which resolution allocations are worth their cost.
That opportunity is a hypothesis to test, not an assumed advantage of ML.

The first decision is whether numerical mesh search can beat the deterministic
baseline consistently. If it cannot, large-scale CNN training is unlikely to help
within this mesh family. Preserve that as a useful negative result and investigate
boundary accuracy, search quality, or the mesh representation before scaling up.

The proposed first scope is vacuum/PEC, 2D TMz (`Ez, Hx, Hy`), a plane wave incident
along +x, one evaluation frequency, and the complete bistatic angular pattern.
Begin with objects roughly 0.5–2 wavelengths across. Use a smooth pulse with useful
energy at the evaluation frequency; select mesh spacing using its significant
bandwidth, not just that frequency. Add oblique incidence and larger electrical
sizes only after the normal-incidence pipeline is qualified.

A 2D silhouette represents the cross-section of an object invariant along z.
An aircraft-shaped cross-section is therefore a synthetic scattering benchmark,
not a prediction of a real 3D aircraft's radar signature. Report **2D scattering
width**, preferably `sigma_2D/lambda`, rather than 3D RCS in square metres.
The normalization follows [Schneider, Chapter 14](https://eecs.wsu.edu/~schneidj/ufdtd/chap14.pdf).

## What the repository actually provides

| Component | Finding | Restart decision |
|---|---|---|
| `solver/coefficients.py`, `solver/cuda/*`, `solver/cuda_runtime.pyx` | Nonuniform TMz updates, CFL checks, float32/float64, GPU stepping | Preserve; extend for incident-field corrections and monitors |
| `solver/reference_tmz.py` | Small NumPy implementation used as a test oracle | Preserve for tiny regression cases; it is not an independent physical reference |
| `mesh.py`, `mesh_projection.py` | Separate x/y coordinate arrays, exact budgets, anchors, min/max spacing, adjacent ratio at most 1.4, fixed PML collars | Preserve the grid representation and constraint layer |
| `scene.py` | Continuous circles and polygons, ordered material overlays, PEC lines | Reuse primitives; add explicit PEC components/holes and topology checks |
| `pml.py`, `sampling.py` | CFS-CPML, physical-coordinate sampling | Preserve and qualify for scattering experiments |
| `simulation.py` | CUDA-only public stepping; point/line additive sources; Ez-only histories | Extend; neither the existing line source nor Ez-only probes constitute a scattering pipeline |
| `ml.py` | Small FiLM-conditioned ResU-Net with two density maps and x/y pooling | Reuse architecture/interface ideas; replace channels, conditioning, and checkpoints |
| `training.py`, `physics.py`, `evaluation/*` | Hashes, resumability, constrained targets, candidate search, Pareto selection, reference gates | Extract small reusable pieces; replace dielectric-specific semantics and receiver-error objectives |
| Legacy generators, stage scripts, trained weights and reports | Experiments around mixed materials and effective-index/edge priors | Archive as historical work; exclude from the new dataset and default workflow |

Three constraints determine the design:

1. The grid is `x[i] × y[j]`. Refining one feature adds resolution along entire
   rows or columns. A proposed 2D patch refinement is not local AMR. The existing
   model already reduces its maps to `rho_x(x)` and `rho_y(y)`; retain this contract.
2. The PEC treatment is point sampling plus `Ez=0` at masked nodes. There are no
   cut-edge lengths, cut-cell areas, or conformal boundary updates. Curved boundaries
   are staircased. Nonuniform spacing does not make them conformal.
3. No TFSF plane-wave injection or near-to-far transformation was found. These are
   new solver capabilities, not data-processing details.

The old work contains useful lessons. The recorded Stage 5 IID experiment reports
median maximum receiver error of 4.680% for uniform, 4.223% for the heuristic, and
4.068% for the selected model. This is evidence of modest improvements in that
experiment, not evidence of PEC scattering performance. Preserve
`docs/stage5_training.md` and the convergence/failure records rather than discarding
the lessons with the old training pipeline.

Local audit: NVIDIA RTX 4070 Laptop GPU, 8188 MiB VRAM, Python 3.12 environment.
The following existing tests produced **42 passed, 30 skipped** in 17.34 seconds:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_mesh.py tests/test_scene.py tests/test_stage2_mesh.py tests/test_cpml.py tests/test_cuda.py -q
```

The CUDA cases were skipped because the native extension is unavailable in this
checkout. Previous GPU validation in `PROGRESS.md` is historical; it has not been
reproduced in this audit. CPU and RAM specifications were not established.

## Staged implementation and acceptance gates

| Stage | Concrete deliverable | Gate before proceeding |
|---|---|---|
| 0. Preserve and simplify | Recoverable legacy snapshot; small solver package; reproducible laptop environment | Existing numerical and mesh tests run, including rebuilt CUDA tests |
| 1. Scattering foundation | Plane-wave injection, E/H contour DFTs, near-to-far transform, analytic cylinder comparison | Incident field, normalization, boundaries, time window, and scattering pattern validated |
| 2. Boundary fidelity | Staircase diagnostic report and, if needed, a qualified conformal PEC implementation | Geometric error is controlled at affordable resolutions before freezing labels |
| 3. Deterministic headroom experiment | Uniform, geometry-rule, and searched meshes for a small synthetic suite | Searched meshes improve the accuracy/cost tradeoff beyond reference uncertainty |
| 4. Laptop CNN pilot | Small reproducible dataset, PEC ResU-Net, held-out FDTD evaluation | Learned meshes recover some search benefit on unseen base geometries |
| 5. Server scaling | Portable build, resumable sharded references/search, larger training set | Laptop/server numerical agreement and measured generation cost are acceptable |

### Stage 0: preserve, then reduce the active project

Before removing files, preserve the current commit under a descriptive legacy tag
or branch, record any uncommitted work, and separately preserve wanted ignored
artifacts/checkpoints. Git does not preserve ignored artifacts. Start implementation
on `codex/pec-scattering-restart`; branch creation and cleanup are future work.

Keep the solver, mesh projector, scene primitives, constants, source utilities,
PML, sampling, build scripts, and their meaningful numerical tests. Retain the
reusable ResU-Net blocks and experiment bookkeeping in small modules. Remove the
legacy experiment code from the active package only after its replacements and
import dependencies are identified. A historical branch avoids maintaining two
active pipelines in the same source tree.

Rebuild through `scripts/build_cuda.ps1`, rerun CPU/GPU parity, cavity convergence,
PEC shielding and CPML checks, and record a small timing/memory baseline. Keep
NumPy stepping available explicitly for tiny tests; a new public CPU runner is
optional because this laptop already has CUDA hardware. Core geometry/meshing
imports should not require loading PyTorch.

### Stage 1: make scattering a trustworthy observable

Implement a normal-incidence TFSF boundary with incident E and H corrections at
their actual staggered positions and times. An auxiliary 1D incident calculation
using the same x grid and timestep is a practical first route. Do not assume an
analytic continuum wave cancels exactly on a graded discrete grid. Validate the
incident amplitude/phase through the object region as well as leakage outside
TFSF. [Schneider, Chapter 8](https://eecs.wsu.edu/~schneidj/ufdtd/chap8.pdf) provides
the standard TMz TFSF construction; the nonuniform implementation needs its own checks.

Add a closed rectangular observation contour in homogeneous vacuum outside the
TFSF region and inside PML. Accumulate `Ez` and tangential H phasors online, using
their actual time coordinates, spatial interpolation, outward normals and
nonuniform quadrature weights. Use one frequency first. Do not record every
contour field at every timestep: the existing history allocation scales with
`Nt × Nmonitor` and would become expensive during reference refinement.

Implement the 2D near-to-far integral and incident-field normalization. Perform an
empty-domain run for each candidate mesh/dt/source configuration during validation;
use matching incident calibration and, where needed, complex-field subtraction
before forming intensity. An empty-domain run is not interchangeable across
different grids. Subtraction cannot compensate for a badly distorted incident
wave at the object, so retain the interior incident-field check.

Validate these cases in order:

- Empty domain: negligible residual scattering and accurate incident field.
- Circular PEC cylinder: independent cylindrical-harmonic series, matched TMz
  convention, convergence with both series truncation and FDTD refinement.
- Rectangle: symmetry and spatial/time convergence.
- Two separated PEC objects and a slotted body: convergence with interaction and
  a resolvable free-space gap.

For the cylinder, compare absolute normalization, angular shape and selected
complex amplitudes, not only normalized curves. Move the integration contour,
increase the air buffer/PML resolution, and extend the time window independently.
All should change the result much less than the proposed candidate error limit.
Use single and double precision comparisons to identify roundoff sensitivity.

### Stage 2: separate boundary error from mesh allocation

Start Stage 1 with the existing staircase solver to minimize simultaneous changes.
For curved and rotated PEC objects, measure whether boundary staircasing dominates
the error and label variability. Include small translations relative to the grid;
mesh search must not merely exploit accidental staircase alignment.

If affordable staircase meshes cannot provide stable reference patterns and useful
headroom, make a conformal PEC update a separate numerical milestone before mass
label generation. Select a published formulation compatible with this TMz
staggering, work out its nonuniform coefficients and stability condition, implement
it first in NumPy, and then port it to CUDA. A 3D conformal recipe should not be
copied into this polarization without deriving the relevant reduction.

Cut-cell stability can impose restrictions beyond the present Cartesian CFL
bound; a blanket multiplier of 0.95 is not a substitute for that analysis. See the
[Benkler, Chavannes and Kuster conformal PEC study](https://speag.swiss/assets/downloads/publications/ursi2006_benkler_fdtd.pdf)
for the connection between geometry, modified updates and stability. Require
small-cut-cell tests, long-time bounded fields, CPU/GPU parity and cylinder
convergence before accepting the implementation.

Freeze the boundary method and its parameters before producing a training dataset.
Any later numerical-method change invalidates the corresponding references and
labels. If the project stays with staircasing, describe it accurately and limit
claims to that discretization.

### Stage 3: prove that a better allocation exists

Generate controlled shapes from continuous geometry: circle, ellipse, rectangle,
wedge, paired conductors with variable gap, and a slotted/U-shaped body. Use a
finite PEC thickness for thin members; treat ideal zero-thickness screens as a
separate later model. Slot and gap effects depend on polarization, so do not assume
all narrow features have equal scattering significance in TMz.

First complete one cylinder validation and one gap/slot example. Then use roughly
12–20 base geometries, one incidence direction, 1–2 electrical sizes, and 2–3
candidate budgets. These are pilot caps, not a statistical generalization claim.
Start reference sweeps around 20/40/80 cells per wavelength and refine further
where needed; no fixed cells-per-wavelength choice guarantees convergence at a
corner or gap.

Establish three baselines using the same continuous geometry and physics:

| Baseline | Purpose |
|---|---|
| Uniform interior | Cost/accuracy curve without learned allocation |
| Geometry rules | Boundary proximity, minimum feature/gap size, corners/curvature, wavelength ceiling |
| Solver-evaluated search | Estimate the available benefit within the same grid family |

Use identical fixed physical PML collars for candidate comparisons. A uniform
interior may need graded transitions into those collars; label it accordingly.
During reference refinement, refine the collars as required while preserving
their physical thickness. Freeze physical source and contour positions. Keep
geometry, significant source bandwidth, observation angles, and physical duration
consistent across candidate meshes.

Tune the geometry baseline on development cases. Enforce a safe wavelength limit,
mesh grading, and preservation of specified gaps/thin members for every method.
The present projector has global axis spacing constraints, not complete local
feature-coverage guarantees: add post-projection topology/coverage checks and
either supported interval constraints or explicit rejection/refinement. Do not
silently fill holes, merge objects, or delete gaps to fit a budget.

Represent search variables as 8–16 coarse density coefficients per axis, then
project them through the real mesher. Begin with uniform and geometry-rule seeds,
bounded coordinate changes, and a few deterministic multi-start perturbations.
Cap the first search at roughly 24–48 evaluated meshes per case/budget. Log the
actual count and cost; extend the cap only when measured results justify it.

Every trial must rerun scattering on the resulting legal grid. A local density
change can move distant lines and reduce the global timestep. Treat its measured
benefit as a property of that complete mesh change, not a local error derivative.
Retain negative improvements and projection failures. Interactions and geometric
aliasing make greedy gains non-additive and non-monotonic. The result is a
**best-found mesh**, not a certified optimal mesh.

Store the evaluated Pareto set, exact x/y coordinates, rebinned realized density
profiles, scattering pattern/complex far field, runtime, cell updates, minimum
spacing, timestep, reference identity and all failures. Near-equivalent meshes
need not have identical density profiles; keep alternatives instead of treating
one arbitrary map as unique ground truth.

Proceed to training only if the searched meshes outperform the tuned geometry
baseline on several geometries with a reproducible margin. A useful provisional
target is at least 20% less work at the same accepted error on a meaningful subset,
with no systematic collapse on the remaining cases. This is a project decision
threshold, not a predicted result; declare it before inspecting the full pilot.

### Stage 4: a small PEC ResU-Net

Use this interface:

```text
continuous PEC geometry + excitation + requested Nx, Ny
    -> raster features and global conditioning
    -> small ResU-Net
    -> positive rho_x and rho_y
    -> deterministic constrained projection and feature checks
    -> x/y grid -> FDTD -> scattering width, error and measured cost
```

Begin with a 128×128 input and width 8 or 16, subject to raster feature-resolution
checks. Input resolution is independent of the simulation grid. If a physical gap
is lost in the input, increase representation resolution or exclude the case with
an explicit reason. More channels cannot restore missing topology.

Start with mask and clipped signed distance `phi/lambda`; use conditioning for
`Lx/lambda`, `Ly/lambda`, `Nx`, `Ny`. When incidence varies, include both
`sin(theta_i)` and `cos(theta_i)` with a documented propagation-direction convention.
Add gap/thickness and curvature channels in ablations, not all at once.

Compute distances in physical coordinates. A distance to the nearest surface is
not by itself a gap-width estimator: use opposing-boundary/medial-axis information
or exact generator metadata for controlled shapes. Compute curvature from a
well-defined contour/smoothing scale and clip corner singularities. Optional signed
normals or `n dot k` preserve directional information; `abs(n dot k)` alone cannot
distinguish front from back. Keep fixed boundary/anchor masks available when the
domain layout becomes variable.

Preserve the two-map/axis-pooling design initially. Direct 1D density heads can be
an ablation later. A four-class 2D refinement mask would suggest unsupported local
refinement and discard the existing natural interface.

Train on realized density/CDF targets from the evaluated meshes, with an optional
projection-consistency loss. Include requested budget in the model input. Select
checkpoints using held-out FDTD error/cost measurements, not just imitation loss.
Use a few budgets to construct a tradeoff curve; do not promise that one inferred
mesh is guaranteed to meet an arbitrary error tolerance.

Do not introduce reinforcement learning, differentiable FDTD, or a large network
in the pilot. Small supervised distillation is sufficient to test whether useful
allocation patterns generalize. The current MILP projector and native FDTD runner
are not an end-to-end differentiable training path.

Split by original geometry/parent family before augmenting. All rotations,
reflections, deformations and electrical sizes from a parent belong to one split.
Hold out entire shape families for a separate harder test. Rotations/reflections
must transform incidence consistently, and any transformed teacher mesh must be
reevaluated because the Cartesian grid changes its relationship to the boundary.
Use a pilot of roughly 50–100 base shapes only after Stage 3 is successful; smaller
sets are sufficient to debug training and deliberate overfitting.

### Stage 5: scale generation before scaling the network

Move reference generation and mesh search to the server first; these are likely
to dominate cost, and their cost must be measured on the laptop. For one physical
case, a rough estimate is:

```text
reference refinements + candidate_count × (scattering run + needed empty calibration)
```

Measure one representative case of each family, including meshing time, before
choosing dataset size. Do not assume a particular training duration or job count
from the GPU model alone.

Use the same case files and command interface on Windows and Linux. Keep paths
relative to an experiment root; store seeds, geometry lineage, solver/boundary
version, mesh-policy version, source/monitor definitions, precision and hashes.
Write results atomically per case and support resume and independent shards.
Preserve failed, nonconverged, timed-out and infeasible cases in coverage reports.

The existing Linux build path is documented but not validated. Parameterize CUDA
architecture targets for the actual server GPU, rebuild native extensions there,
and run a small frozen laptop/server parity suite before any long job. Reuse the
Python dependency lock where compatible; record platform-specific changes.
Use one process per GPU initially and cap CPU meshing threads to avoid oversubscription.
Scheduler integration follows the server environment; do not assume Slurm.

## Metrics and reference qualification

Use a predefined full angular grid (for example 0–359 degrees at one-degree steps,
then verify angular convergence) and linear scattering width:

```text
E_sigma = ||sigma_candidate - sigma_reference||_2 / ||sigma_reference||_2
C_updates = Nx * Ny * ceil(T_physical / dt)
T_total = feature construction + inference + mesh projection + FDTD + far-field processing
```

Declare a numerical floor/absolute-error alternative for nearly zero-scattering
cases. Retain complex far-field error, selected backscatter error, and dB plots
with an explicit display floor. A small integrated pattern error alone does not
guarantee accuracy at a deep null. Keep physical phase origin and normalization fixed.

Start with a provisional 5% candidate pattern-error target and approximately 1%
reference uncertainty/change target. Require at least two successive refinement
comparisons below the reference threshold plus independent time/PML/contour checks.
For the cylinder, require agreement with the analytic solution as well. Passing
successive-grid checks is empirical qualification, not proof of exactness.
If those targets cannot be demonstrated, report nonconvergence and investigate
the cause; do not weaken the gate silently to increase dataset size.

Account for **both** cell count and timestep. For the current vacuum Cartesian
scheme, the timestep is bounded by
`1 / (c0 * sqrt(dx_min^-2 + dy_min^-2))`; conformal changes need their own bound.
At equal Nx/Ny, one tiny interval can still make a mesh slower. Count the full
rectangular computational grid, including PEC interior nodes that the current
runtime still stores and visits. Include empty calibration cost when required by
the deployed method. Report offline reference/search/training cost separately.

Primary plots: error versus total runtime and cell updates; achieved tolerance
versus cost; and failure/coverage rate. Compare each method at equal accuracy as
well as equal budgets. Report median, upper-tail and worst-case behavior by
geometry family, including budget-infeasible cases. Benchmark inference and mesh
projection separately: the existing MILP's overhead may erase a small FDTD saving.

If a predicted mesh is infeasible, use an explicitly reported deterministic
fallback. Mesh validity and confidence do not certify scattering accuracy; assess
that on held-out solver evaluations. Test cases must not influence teacher search
settings, baseline tuning, checkpoint selection or stopping criteria.

## Proposed active layout and first work package

Keep the package name `fdtdmesh` and introduce capabilities incrementally:

```text
src/fdtdmesh/
  solver/                 existing stepping, incident corrections, contour DFTs
  mesh.py, mesh_projection.py, pml.py, sampling.py, constants.py
  geometry/               PEC shapes, holes, feature checks, raster channels
  scattering/             case contract, incident wave, monitors, far field, cylinder series
  meshing/                geometric baseline and post-projection feature validation
  learning/               reusable ResU-Net blocks, PEC model, dataset, training
  experiments/            references, bounded search, evaluation and CLI
configs/                  laptop smoke, pilot, server generation
examples/                 cylinder, gap, slot and mesh comparison
tests/                    core numerics, scattering, geometry, data lineage, small learning tests
artifacts/pec/            ignored run outputs, references and checkpoints
```

Prefer small case/result objects such as `PECScatteringCase`, `MeshRequest` and
`ScatteringResult`. Geometry, source, observation angles and physical run duration
belong to the case; candidate density/budget belongs to the mesh request. Results
carry complex far fields, normalized scattering width, mesh diagnostics, timings,
and explicit status. A reference identity must include all physics and numerical
choices that can change the observable.

The first implementation work package ends with **one validated PEC cylinder
scattering example on uniform and graded grids**, its analytic comparison,
convergence plots and machine-readable metrics. It includes restoring the CUDA
build, retaining core tests, a minimal scattering case contract, normal-incidence
TFSF and contour/far-field extraction. It does not require a trained CNN or a large
image collection. The next package adds the gap/slot headroom experiment.

After that, introduce complex synthetic polygons and then selected silhouettes
with known provenance and explicit physical scale. Preserve intentional holes,
gaps and disconnected pieces during contour cleanup. Photographic segmentation
quality is a separate uncertainty and should not obscure the first numerical test.
