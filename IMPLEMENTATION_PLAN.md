# Learned FDTD meshing implementation plan

Updated: 2026-09-19.

This document consolidates the referenced **Mesh Training Strategy** into an
implementation roadmap. It distinguishes the first-stage deliverables from later
training and physics work. See [PROGRESS.md](PROGRESS.md) for implementation status,
[README.md](README.md) for the current API, and
[docs/validation.md](docs/validation.md) for measured validation.

**Current priority:** prove the method with the original width-16 CNN on richer
geometry/material distributions and better-qualified fine-grid references. Larger
CNNs are deferred until this proof of concept passes physical validation. Section
11 specifies the next implementation; its generator-v5 settings and reference
limits are proposals, not features or results already delivered.

## 1. Objective

Build a research framework that learns where to allocate an exact number of
nonuniform tensor-product Yee cells for a continuous electromagnetic scene.
The eventual objective is lower electromagnetic error at the same simulation
cost, or lower cost at the same error, compared with uniform and heuristic meshes.
Teacher agreement is a starting point, not the final measure of success.

The pipeline is:

```text
Continuous scene + material/source/receiver definitions
    -> fixed-resolution raw-physics raster + global conditioning
    -> ResU-Net x/y importance logits
    -> positive one-dimensional axis densities
    -> deterministic constrained mesher
    -> nonuniform CUDA TMz simulation
    -> electromagnetic error and computational cost
```

The CNN determines where resolution has value. The deterministic mesher owns
cell counts, anchors, boundaries, monotonicity, spacing, and grading constraints.

## 2. Contracts that persist across stages

- Geometry is stored in continuous physical coordinates, independently of any
  simulation or CNN grid. All coordinates are metres; integers are not grid indices.
- Preserve the familiar `FDTD_2D_Ez`, material, geometry, source, monitor, `config`,
  and `run` style of the previous library at
  `C:\Users\Traveler\PycharmProjects\FDTD`. Use that library as a reference rather
  than copying it wholesale or changing it as part of this project.
- `Nx` and `Ny` are exact total cell counts. Final axes have `Nx+1` and `Ny+1`
  lines. Anchors consume the existing budget and must never be silently dropped.
- The same authoritative scene must support CNN rasterization, candidate meshes,
  and increasingly fine uniform reference meshes.
- Compare simulations at a fixed physical duration `T`, with
  `Nt = ceil(T / dt)`. Report `Nx * Ny * Nt`, GPU time, and electromagnetic error
  separately. Tiny cells reduce the CFL timestep and therefore increase work.
- Production stepping stays in native CUDA. Allocate/upload before stepping and
  download requested outputs afterward. No Python timestep loops, source callbacks,
  monitor callbacks, or host/device transfers inside the time loop.
- NumPy stepping is a test oracle, not an automatic production fallback.
- Do not differentiate through integer mesh allocation and CUDA FDTD in the initial
  training workflow. Improve target densities through evaluated candidate meshes.

## 3. Repository organization

The first stage uses a small module layout. Split modules as later stages need
additional responsibilities; avoid creating empty scaffolding for unimplemented work.

```text
src/fdtdmesh/
    simulation.py             Public scene-to-mesh-to-run lifecycle and results
    scene.py                  Continuous materials, geometry, anchors, probes
    sources.py                Vectorized source waveform preparation
    mesh.py                   Exact-budget density meshing and constraints
    mesh_projection.py        Joint integer anchor assignment and L1 projection
    pml.py                    Fixed collars and staggered CFS-CPML profiles
    sampling.py               Physical probe interpolation and dual-cell areas
    ml.py                     Rasterization, ResU-Net, conditioning, model I/O
    data/                     Versioned scene schema, seeded generation, split checks, CLI
    evaluation/               Common observables, refinement, candidate scoring and reports
    solver/
        coefficients.py       Staggered material coefficients and CFL limit
        reference_tmz.py      NumPy testing oracle
        tmz.py                CUDA backend discovery
        cuda_runtime.pyx      Validated Cython/native boundary
        cuda/                 C++ API and nvcc-compiled device/runtime code
tests/                        Mesh, geometry, model, numerical and integration tests
examples/                     Uniform, nonuniform and checkpoint-driven simulations
benchmarks/                   Fixed-duration performance experiments
scripts/                      Build and, later, dataset/training/evaluation commands
docs/                         Numerical conventions and validation records
```

Stage 3 adds `data/` and `evaluation/`. As training develops, separate `ml.py`
into a package when model, losses, datasets, and training warrant it. Keep a `uv`
environment and dependency lockfile, and commit coherent changes to Git.

## 4. Stage 1 — solver and CNN-to-FDTD mesher

### Task A: nonuniform CUDA TMz solver

Implement only `Ez`, `Hx`, and `Hy` on an arbitrary tensor-product x/y grid.

1. Implement continuous isotropic, nondispersive materials: positive `epsilon_r`,
   optional positive `mu_r`, nonnegative electric conductivity, vacuum, and exact PEC.
2. Support rectangles, circles, triangles, simple polygons, and axis-aligned thin
   PEC lines. Preserve insertion priority and explicit anchors.
3. Use `Ez (Nx+1, Ny+1)` at nodes, `Hx (Nx+1, Ny)` and `Hy (Nx, Ny+1)` on staggered
   edges. Use primal widths in magnetic updates and dual widths in electric updates.
4. Precompute update coefficients, including centered electric conductivity and
   a conservative nonuniform CFL bound. Support float32 and float64; reject unsafe dt.
5. Implement PEC outer boundaries, soft point/line sources, Gaussian and sinusoidal
   waveforms, point/line Ez receivers, final fields, and runtime/work diagnostics.
6. Compile `.cu` files with nvcc and expose a C/C++ API through Cython. Provide a
   repeatable Windows MSVC/nvcc build. Release the GIL during the native run and
   release device resources on success or failure.
7. Keep all stepping data on the device. Precompute source waveforms and source
   overlap sums before upload; record receivers without Python intervention.
8. Keep the public lifecycle scene-first: uniform meshing, supplied coordinates,
   density meshing, or model meshing, then `run()`. Revalidate anchors before running.

Acceptance criteria:

- CUDA agrees with an explicit NumPy oracle for uniform/nonuniform meshes in both
  precisions, including conductivity, magnetic materials, sources, and receivers.
- Independent analytical cavity tests demonstrate convergence and check signs,
  staggering, dual widths, and lossy material updates.
- PEC nodes remain zero and an anchored PEC wall blocks transmission.
- Repeated runs are reproducible; empty source/receiver cases work; invalid inputs
  fail before unsafe native access.
- Residency accounting and runtime source inspection show no stepping transfers.
- Work comparisons use the same physical simulation duration.

### Task B: CNN-to-FDTD mesher

1. Rasterize raw physical inputs: permittivity, normalized conductivity, PEC,
   sources, receivers, x/y anchors, and permeability when supported. Do not start
   with handcrafted distance, gradient, edge, curvature, or gap channels.
2. Condition on electrical domain size, frequency ratio, and x/y budgets. The
   implemented contract uses `log(Lx*f_max/c0)`, `log(Ly*f_max/c0)`, `f_min/f_max`,
   `log(Nx)`, and `log(Ny)`.
3. Implement a fully convolutional residual U-Net with FiLM at multiple scales.
   Predict two logits fields, pool over orthogonal axes, then apply softplus plus
   a positive floor. Use normalized log-sum-exp (log-mean-exp) to avoid an artificial
   dependence on raster size for constant predictions.
4. Treat densities as piecewise constant in raster bins and integrate quantiles
   over the learned interior; fixed collars ignore CNN density.
5. Jointly optimize integer anchor assignments and line positions to minimize
   normalized mean absolute quantile-line displacement. Enforce mandatory adjacent
   ratio <= 1.4, optional stricter ratio and min/max widths, exact anchors, counts,
   and collar lines. Distinguish solver timeout from proven infeasibility.
6. Verify exact counts, exact boundaries and anchors, strict monotonicity, finite
   coordinates, and deterministic output. Reject infeasible requests explicitly.
7. Save self-describing checkpoints: format version, architecture, weights, ordered
   channels, normalization, raster dimensions, conditioning, pooling, output semantics,
   training commit, and dataset version. Reject incompatible contracts.
8. Expose `load_mesh_model()` and `mesh_with_model()`, including explicit budget
   overrides. Provide a full inference-to-CUDA example, clearly distinguishing
   random demonstration weights from trained weights.

Acceptance criteria:

- Exact budgets and all anchors survive deterministic meshing, including randomized
  density/anchor cases and spacing/grading projection tests.
- Tests distinguish x/y pooling orientation and verify positive output and gradient
  flow through the network.
- Electrically equivalent scaled scenes have consistent conditioning and raster maps.
- Checkpoint round-trip preserves predictions; incompatible metadata is rejected.
- A loaded CNN checkpoint produces a legal mesh that runs on the CUDA solver.

Historical stage-one exclusions (CPML is now added in stage 2): CPML, TEz, dispersion, anisotropy, PMC, periodic boundaries,
TF/SF, waveguide eigensolvers, NF2FF, 3D, plotting inside the solver, trained model
quality claims, and a CPU production backend. Training is a later stage.

## 5. Stage 2 — open boundaries and solver readiness for datasets

Implemented after PEC/nonuniform validation. See [stage 2 evidence](docs/stage2_validation.md).

1. Reserve a fixed number of PML cells per side **inside the total Nx/Ny budget**.
   Define physical collar thickness and reject budgets with no feasible interior.
2. Keep PML spacing uniform in its normal direction; permit tangential nonuniformity.
   Fix PML internal lines and interfaces independently of CNN predictions.
3. Expose PML interfaces as hard anchors in both the raster and mesher. Enforce
   grading at the interior/collar transition. Keep geometry, sources, and receivers
   out of PML in this first CPML implementation.
4. Implement staggered CFS-CPML coefficients and device-resident auxiliary state.
   Preserve the existing no-transfer runtime contract.
5. Test attenuation/reflection against an enlarged-domain reference, normal and
   oblique propagation, corners, uniform/nonuniform interiors, and long-run stability.
   Establish quantitative tolerances before accepting this stage.

6. Use physical point current deposition normalized by dual-cell area and dt;
   retain explicit legacy field-increment and per-node current-density modes.
7. Interpolate receivers at fixed physical coordinates; provide fixed sample counts
   for lines and common-time resampling without extrapolation. Define DFT/window
   conventions for fair comparisons.
8. Add the raw PML channel and version-2 checkpoint meshing policy. Record projection
   correction metrics and provide a detached repaired-CDF auxiliary training loss.

Acceptance: <1% normalized waveform peak/L2 and final interior error against the
same mesh extended outward, for 0/30/45/90-degree vacuum packets on uniform and
nonuniform grids; finite late fields with <1% initial norm after 1.5 ns at 45 degrees.
PEC controls must produce >10% peak error, confirming sensitivity to reflection.
Also require CUDA/NumPy CPML agreement in both precisions, zero stepping transfers,
exact fixed collars, current conservation, and physical probe interpolation tests.
These tests qualify the implemented cases, not arbitrary broadband grazing incidence.

Device-side DFT, energy diagnostics, bounded recording, graph replay, and batching
remain optional later work driven by profiling and dataset requirements.

## 6. Stage 3 — procedural scenes, trusted references, and evaluation

Implemented in version 0.3.0 and expanded in 0.4.0/0.4.1. See the
[original stage 3 evidence](docs/stage3_validation.md) and
[broader generator / ring-down contract](docs/dataset_v2.md), and
[dk mixture and randomized probe validation](docs/dataset_v3.md).
The pipeline is complete; individual scenes remain unusable as references until they
pass the recorded convergence checks. The initial validation accepts 15/32 scenes.

1. Create versioned scene/data schemas containing physical scene definitions,
   random seeds, budgets, frequencies, source/receiver conventions, mesh parameters,
   solver precision, boundary conditions, and code provenance.
2. Generate mixed scenes with dielectric and PEC rectangles, circles, triangles,
   polygons, thin lines, gaps, touching objects, and several geometric scales.
   Vary object count (1–8 ordinarily, 9–12 held out), independent object materials,
   logarithmically distributed sizes/contrast/loss, domain aspect ratio/electrical
   size, object orientation and vacuum probe locations. Use 128x128 rasters to
   resolve smaller features. Keep generation configuration and versions in manifests.
   Generator v3 draws dk from a configurable 85% log-uniform 1–10 core and 15%
   log-uniform 10–30 tail; material-OOD remains 36–120. Check spatial coverage for
   the source and every receiver index independently to guard against fixed-location
   shortcuts. This describes the historical v3/v4 distribution. The next generator
   replaces the mandatory high-dk tail with the richer in-range material and
   contact/overlap distributions in section 11.
3. Require important unanchored features to span several CNN pixels initially.
   If necessary later, add raw subpixel samples/material fractions rather than
   semantic geometry features.
4. Generate fine uniform references and refine until successive observables agree
   within a recorded convergence tolerance. Retain analytical cases as independent
   checks, and mark nonconverged scenes rather than treating them as ground truth.
   Separately test late-time receiver ring-down after excitation ends; extend the
   physical duration and restart spatial checks when needed. Preserve temporal and
   frequency sampling resolution and compare every candidate over the accepted
   shared window. Longer simulation time does not substitute for spatial refinement.
5. Compare receiver waveforms on a common physical time basis, requested spectra,
   and phase where meaningful. Use normalization floors and document weighting.
6. Separate IID, compositional, geometry-OOD, material-OOD, scale-OOD, and unseen-budget
   evaluations. Prevent near-duplicate scenes from crossing dataset splits.
7. Establish uniform, heuristic, and CNN baselines with identical scene, excitation,
   physical duration, observable definitions, and error metrics.
8. Use a deterministic 64/16/32 train/validation/IID reference manifest for the first
   physics-target cycle. Generate references per scene with atomic status commits,
   exact-configuration resume checks, incremental coverage reports, and no silent
   removal of failed or unconverged cases. Keep the IID test split untouched during
   search and distillation.

Acceptance: reproducible scene generation, documented reference convergence,
split integrity checks, and an accuracy-versus-work/runtime report for the baselines.
The production reference corpus is accepted scene by scene; creating its manifest or
finishing only part of a resumable sweep does not imply that all 112 labels converged.

## 7. Stage 4 — teacher imitation

Implemented in version 0.5.0. See [measured Stage-4 evidence](docs/stage4_training.md).
The trained model strongly reduces teacher-density loss and improves median physical
teacher agreement, while a sensitive held-out outlier demonstrates why Stage 5 needs
physics-selected targets.

1. Implement or adapt the heuristic teacher to produce axis density targets compatible
   with the new exact-budget mesher. Do not assume an old checkpoint matches the
   current architecture or raster contract.
2. Train the conditioned ResU-Net over multiple scenes, electrical sizes, and budgets.
   Normalize density targets consistently because a global density scale does not
   change the unconstrained quantile mesh.
3. Add the detached repair loss with a modest weight and monitor projection
   correction metrics alongside physical quality. Add reproducible training/resume commands, validation, seed/config logging, and
   checkpoints with actual dataset and Git provenance.
4. Evaluate generated meshes through the deterministic mesher and real CUDA solver,
   rather than relying only on a density imitation loss.

Acceptance: a genuinely trained, loadable checkpoint; held-out imitation and physical
evaluation results; and stable constraint satisfaction across the supported budgets.

Measured acceptance: the epoch-40 checkpoint records dataset/Git provenance; all 64
IID scene-budget projections are legal; held-out CDF loss is 14.9x below the seeded
untrained control; and 16 real-CUDA comparisons are recorded. This accepts the
imitation pipeline, not electromagnetic improvement over the heuristic.

## 8. Stage 5 — physics-generated targets and distillation

The first substantial cycle is implemented and measured. See
[Stage-5 evidence](docs/stage5_training.md). Further iterations can expand candidate
families and reduce the retained Stage 4 prior.

1. For each scene/budget, propose uniform, heuristic, current CNN, perturbed-density,
   and uniform/adaptive-mixture candidates.
2. Generate legal meshes and run every candidate for the same physical duration.
   Reject invalid/nonfinite runs and retain diagnostics describing the failure.
3. Score physical error and cost. A scalar `J = E_EM + beta*C_normalized` can guide
   selection, but also retain the error/cost Pareto frontier and its underlying data.
4. Distill improved candidate densities into the CNN. Start with teacher supervision
   as a strong prior, then reduce its weight as physics-generated targets improve.
5. Repeat candidate search, real-FDTD evaluation, and distillation. Keep an untouched
   test set and compare to both the original teacher and the imitation checkpoint.
6. Use feasible low-budget strata at 32x32 and 48x48 alongside 64x64 and 96x96.
   Record infeasible scene-budget pairs, balance sampling across budgets, normalize
   losses within each budget, and report equal-budget accuracy/cost comparisons.

The first four-budget pilot is complete. It reuses fixed converged references, records
infeasible low-budget pairs, and reports validation and held-out FDTD metrics by
budget. It improves several strata but does not yet meet aggregate held-out acceptance,
so the earlier selected checkpoint remains the default while server reference coverage
is expanded.

Acceptance: measured improvement in held-out electromagnetic error at matched cost,
or lower cost at matched error. Teacher agreement alone is not sufficient.
Let the model learn when uniform allocation is adequate; do not hard-code a
budget threshold that forces uniform meshes.

## 9. Stage 6 — generalization, ablations, and optional surrogate

Report accuracy/cost frontiers across the designated OOD splits and unseen budgets.
Study conditioning, pooling, density regularization, grading, raster resolution,
and the transition toward uniform allocation at generous budgets.

If real-FDTD candidate evaluation becomes the bottleneck, investigate a differentiable
surrogate mapping scene, densities, and budget to predicted error/cost. Any surrogate-
optimized mesh must still be validated with real FDTD. Spatial error attribution,
subpixel representations, and TEz are later research extensions, not prerequisites
for the first successful TMz training loop.

## 10. Delivery discipline

For each stage, document the final numerical/API contracts, run relevant tests and
examples, record validation hardware and limitations, update [PROGRESS.md](PROGRESS.md),
and commit the implementation. Keep acceptance evidence separate from planned work.
Avoid claiming training quality, open-boundary accuracy, or GPU performance that
has not been measured.

## 11. Revised next milestone — small CNN, rich scenes, strict fine references

### 11.1 Scope and preserved baseline

Use the original `ResUNet(width=16)` (128,418 parameters), existing FiLM blocks,
nine raw-physics channels, axis-density output and constrained mesher. Start a
fresh proof-of-concept training run on the new dataset; preserve historical
checkpoints as comparators. The remote checkout does not contain the old trained
checkpoint files, so checking out the old source does not restore those weights.
Do not change CNN width/depth while qualifying the new generator and references.

Work on `codex/small-cnn-rich-scenes`, branched from `366508e`. The earlier large-CNN
proposal remains on `codex/server-scale-cnn`; no large architecture had been
implemented. The superseded generator-v4 server sweep was stopped and its artifacts
retained. Do not mix its references into the new dataset without exact scene,
material, observation and acceptance-contract checks.

The server has four independent 24-GiB TITAN RTX GPUs, 20 physical CPU cores,
approximately 109 GiB host RAM, and 8.6 TB free disk at inspection. Use this capacity
first for numerical qualification and more informative data. Larger CNN capacity
is a later controlled experiment, not the immediate objective.

### 11.2 Generator v5: deliberate geometry interactions

The current generator includes one paired, axis-aligned touching-rectangle case,
but rejects general bounding-box overlaps and fixes all ordinary `mu_r` to one.
Replace its global separation rule with explicit, measured interaction classes.
Initial target shares below are design settings to audit after rendering, not
claims about the current generator:

| Scene class | Initial share | Required variations |
|---|---:|---|
| Separated objects / resolved gaps | 20% | Near/far separation, narrow/wide gaps, sparse controls |
| Contact | 25% | Shared edges, tangency, corner/point contact, multi-object junctions |
| Intersection / partial overlap | 25% | Rotated crossings, shallow/deep penetration, mixed shape pairs |
| Nested / layered | 15% | Inclusions, shells, vacuum cutouts, multiple interface depths |
| Mixed assemblies | 15% | Several interaction types, connected clusters, cavity-like arrangements |

Use 1–12 primitives in the ordinary distribution, stratifying sparse and dense
counts so neither dominates. Reserve 13–20 and unseen assembly combinations for
compositional challenges after qualification. Vary shape pairings, translation,
orientation, aspect ratio, size ratios and occupied fraction; keep circle,
rectangle, triangle, polygon and supported axis-aligned PEC-line families. Add
simple concave polygons as a tested extension, not arbitrary self-intersecting
polygons. Vary global domain aspect/electrical size and local feature scales
separately; record realized distributions and generation rejections.

Define geometry in continuous coordinates. Exact contact uses shared construction
coordinates and a documented scale-aware geometric predicate, not a random gap
that happens to disappear on one raster. Audit the *visible composite*: topology,
overlap fraction, exposed interfaces, remaining gaps/necks and material occupancy.
Requested object count is not a substitute for visible complexity. Reject or
resample completely occluded objects and accidental tiny residual fragments with
explicit counts; do not silently call them valid complex scenes.

Retain the existing **last-inserted primitive wins**, including PEC. An overlap
assigns one material at each point; it does not add permittivities, permeabilities
or conductivities. Preserve order in the scene hash. A later vacuum primitive may
form a cutout; a later dielectric may overwrite PEC under the same rule. Test
material/material and PEC/material ordering on raster and all staggered Yee sites.
If physical union semantics are wanted for a scene family, construct and record
that order explicitly rather than changing global priority.

Keep the PML collar vacuum and keep objects, sources and probes out of it. Place
probes using the final material map instead of rejecting the whole bounding box
of an overlapping assembly. Maintain explicit source/probe/interface clearance
for the supported interpolation stencil. Start with vacuum probes and one source;
embedded-material probes are a later interpolation-validation extension.

Keep 128x128 CNN inputs initially and require finite-width visible features, gaps
and necks to span at least four input pixels. Test composite features after overlap,
not only the original primitive extents. Exact zero-width contacts and anchored PEC
lines need explicit topology/anchor tests and their own convergence statistics.
If useful geometry cannot be represented, test 256x256 inputs with the *same*
width-16 CNN as a separate ablation; version raster/target metadata and retrain.
Do not expect finer FDTD labels to recover information absent from CNN inputs.

### 11.3 Material diversity without a compulsory epsilon tail

Use positive, isotropic, nondispersive constitutive parameters within the current
solver contract. The ranges below define synthetic numerical experiments; they
are not broadband constitutive models for particular manufactured materials.

| Quantity | Proposed ordinary distribution | Coverage requirement |
|---|---|---|
| `epsilon_r` | 1–10, stratified continuous draws across 1–2, 2–4, 4–7, 7–10; include exact unity controls | Remove the mandatory 15% >10 tail; cover weak/strong *interface contrasts* within the range |
| `mu_r` | Exact 1 for half of ordinary material draws; remaining draws stratified over 1–4 | Include magnetic-only (`epsilon_r=1`), dielectric-only, and joint variation |
| `sigma_e` | Explicit zero-loss mass, initially 25%; remaining draws stratified by loss ratio described below | Cover low, moderate and strong loss without conflating finite conductivity with PEC |
| PEC | Separate categorical material, initially around 20% of primitives | Audit realized PEC occupancy/connectivity as well as requested primitive fraction |

For nonzero conductivity, stratify the dimensionless ratio
`r = sigma_e / (2*pi*f_ref*EPS0*epsilon_r)` over `1e-4–1e1`, using the recorded
source carrier as `f_ref`, then store the resulting physical `sigma_e` in S/m.
This gives comparable loss regimes across randomized frequencies and epsilon.
Record the physical conductivity range and realized ratio distribution. Include
matched controls with one parameter varied at a time and independently crossed
epsilon/mu/loss bins; avoid tying all magnetic objects to one loss or geometry class.
Any calibrated physical conductivity bounds and resampling bias must be recorded.

Audit refractive-index proxy `sqrt(epsilon_r*mu_r)`, impedance proxy
`sqrt(mu_r/epsilon_r)`, interface contrasts, attenuation lengths and finite-loss
skin depths over the excitation band. Lossless local wavelength alone is not a
sufficient resolution estimate for highly conductive objects. Very thin conductive
layers may need extra refinement or an explicit resource rejection. Do not clip
them to PEC or silently remove difficult material combinations.

Do not automatically retain the previous material-OOD range 36–120. Initially
evaluate unseen joint combinations *within* the new ordinary ranges. Values above
10 for epsilon, mu below 1 or above 4, and more extreme conductivity are separately
versioned optional extrapolation studies after the ordinary proof of concept.

The current raw channels already contain `mu_r` and normalized conductivity; no
additional channel is needed for constant isotropic permeability. Keep the original
normalization for the first comparison and inspect channel scales/optimization.
If a transform becomes necessary, version it and train a separate checkpoint.

### 11.4 Source, scale and distribution coverage

Vary source/probe coordinates, assembly orientation relative to propagation,
source-to-object and receiver-to-object distances, domain aspect, occupied area,
electrical size and bandwidth. Retain the existing physical point-current
normalization. Global amplitude variation alone in this linear solver is not a
meaningful new meshing task under normalized errors.

The current CNN sees binary source/receiver maps and five conditioning values; it
does not receive pulse delay/width or independent per-source phases/amplitudes.
For the first controlled run, use one fixed normalized pulse prescription derived
from the recorded frequency band, varying the band and positions. Before allowing
independent pulse shapes, multiple-source phases or line-source profiles, expose
the distinguishing information in a versioned input/conditioning contract. Do not
train on different physical targets that are indistinguishable to the CNN.

Use independent deterministic streams per split and persistent scene IDs. Adding
training scenes must not shift validation/test scenes, which the current single
generation stream can do when counts change. Hash the ordered resolved scene,
generation configuration, source contract and material priority. Detect leakage
between transformed versions and near-duplicate assemblies, not just exact seeds.

### 11.5 Fine reference policy and resonance qualification

More computation permits stronger evidence; it does not guarantee convergence of
every discontinuous/contact geometry. Retain direct Yee-location material sampling
and qualify its staircase error. Do not introduce subpixel averaging in this work.

Proposed reference policy, to qualify on the numerical pilot before production:

| Setting | Next policy |
|---|---|
| Precision | Float64 reference solver |
| Uniform levels | 128, 256, 512, 1024, 2048, 4096; targeted 8192 only after profiling |
| Minimum accepted level | 1024, plus geometry/wavelength/attenuation resolution checks; implement this new acceptance field explicitly |
| Spatial acceptance | Both per-receiver waveform and complex-spectrum relative errors <=2%, for two consecutive refinements |
| Initial duration | At least 48 band-reference cycles and pulse end + 8 conservative material-transit times; replace the old epsilon-only estimate with one including mu |
| Tail acceptance | Pulse ended before the final 20% window; receiver RMS/peak <=1% |
| Duration extensions | Up to 6 doublings from the new base duration; restart spatial sequence on every extension |
| Work ceiling | Pilot starting cap of 20,000,000,000,000 cell updates per attempted grid/window; log cumulative work per scene too |
| Field/history estimates | Pilot starting ceilings 6 GB field estimate and 1 GB receiver-history estimate per job, subject to measured total CPU/GPU memory |
| Sampling | Maintain at least existing 16 time samples/period and frequency resolution proportional to duration; never silently coarsen on hitting a cap |

Initial duration uses a conservative bound based on material propagation, e.g.
`sqrt(max(epsilon_r)*max(mu_r))*domain_diagonal/C0`, plus the pulse duration.
It is a starting window, not a prediction of resonance lifetime. Material loss,
contact, PEC cavities and multiple scattering require observed ring-down checks.
Keep units/frequency conventions explicit in the manifest; avoid applying a second
unrecorded duration multiplier to the already enlarged base window.

The current field estimate is `240*(N+1)^2` bytes for float64. This is about
1.008 GB at 2048, 4.028 GB at 4096, and 16.110 GB at 8192. The earlier 1 GB cap
therefore forbids even 2048. The user's revised compute scope permits raising
resource ceilings, while convergence tolerances remain unchanged. These estimates
are not full process-memory bounds: profile coefficient construction, temporary
arrays, solver copies, histories and postprocessing on both host and GPU. Qualify
2048 then 4096 on one GPU before scheduling four concurrent jobs. Do not assume an
8192 job fits safely merely because its estimate is below 24 GiB; schedule such
jobs separately only after measured headroom and host-memory checks.

Expose missing controls in the CLI/run identity: history and observation limits,
minimum accepted level, total-work/wall-time budgets and per-level diagnostics.
An initial pilot may raise observation caps to 262145 time samples and 32769
frequencies, but must estimate storage and spectral-computation cost before launch.
If these are insufficient, retain a resource-limited outcome; never reduce sampling
resolution silently. Bounded receiver recording/streaming and restart checkpoints
for individual expensive levels are follow-up infrastructure before unlimited
resonance or 8192 campaigns. Keep final waveform records needed for candidate
comparison, not just a small set of DFT bins.

Avoid false convergence when coarse grids all miss a thin feature. Derive feature,
wavelength and attenuation resolution requirements before accepting the two passes;
record the thresholds calibrated by independent cases. Retain waveform/spectral
floors and report weak-signal cases instead of obtaining tiny errors by normalizing
against an arbitrary large scale. Audit time-window stability on a longer-window
qualification subset using a documented comparison/window convention. Adding time
changes Hann-window spectra, so compare like windows or a separately defined
invariant observable, not incompatible spectra.

Qualification includes magnetic/lossy interface reflection and transmission,
layered media, closed/open cavity resonances, rotated interfaces, PEC/material
ordering, exact contacts, intersection-created corners and narrow necks. Check
CUDA/NumPy agreement and applicable analytical solutions. On representative and
high-error cases test a halved timestep and enlarged PML separation/thickness;
ensure PML or temporal errors are not mistaken for spatial convergence. High-Q or
closed lossless resonators may never satisfy the pulse-tail gate: keep them as
explicit time-unsettled challenge cases, not accepted decayed references.

Preserve all statuses, level/window attempts, resource failures and infeasible
scene-budget pairs. Use a new manifest/run identity and new output roots for v5.
Do not overwrite v4 outcomes or reuse weaker references simply because a scene
name matches. Upgrade/retry records retain the original attempt and full config.

### 11.6 Staged delivery and small-CNN proof of concept

1. **Generator and material correctness.** Implement v5 configuration/schema/CLI,
   contact/overlap construction, visible-feature checks, material distributions,
   mu-aware duration and stable split streams. Test priority, exact contacts,
   deterministic hashes, leakage, probe placement and rendered distribution
   coverage. Produce a gallery and material/topology statistics before a sweep.
2. **Numerical qualification.** Use a separate roughly 160-scene engineering pilot
   balanced over the five interaction classes plus independent analytical fixtures.
   It is a development set, not an untouched test set. Profile 2048/4096 and long
   windows, qualify memory/sampling limits, and inspect convergence curves and
   rejected cases. Freeze the acceptance/configuration contract before production.
3. **Versioned proof-of-concept corpus.** Start with 1,024 train, 128 validation and
   256 untouched IID scenes; extend training toward 2,048 only after accepted
   coverage is measured, preserving holdouts exactly. These are attempted scene
   counts, not promised converged labels. Keep separate contact/material/composition
   challenge evaluations and report rejection rates by stratum.
4. **Width-16 training from scratch.** Generate restart-safe heuristic targets,
   retaining infeasibilities, for budgets 32/48/64/96. Balance sampling and normalize
   losses across budgets; split by scene, never by budget pair. Use teacher
   pretraining then real-FDTD candidate search and physics distillation. Compare
   uniform, heuristic and small CNN over identical accepted windows. A 100-epoch
   cap with validation early stopping is an initial experiment setting, not a
   required training duration. Keep the width-16 architecture fixed.
5. **Physical selection and acceptance.** Select on validation FDTD; report
   per-budget and per-material/topology-stratum median, mean and tail EM errors,
   work, runtime, failure and feasibility. Use paired scene-level uncertainty and
   include acceptance bias/coverage. Predeclare an error-versus-work selection
   criterion and require credible held-out improvement, without hiding important
   stratum regressions. If it fails, diagnose data, targets, meshing and input
   information before enlarging the CNN.
6. **Only then revisit capacity.** Compare width 32/64 against width 16 on the same
   fixed references, splits and search budget. Revisit deeper models only if the
   measured small-model learning curves or physical errors justify them.

This revision is a plan and source-baseline checkout. Generator v5, enlarged
reference limits, new targets and training are not launched by the planning change.
