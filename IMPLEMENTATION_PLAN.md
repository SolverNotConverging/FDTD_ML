# Learned FDTD meshing implementation plan

Updated: 2026-09-20.

2026-09-21 update: the 3,000-scene combined corpus is complete. The next teacher/CNN
workflow uses eight budget pairs per training scene, four extra reserved validation
pairs, 2× sparse sampling and budgets 48–128 on each axis. See
[mixed-budget training](docs/training_mixed_v7.md) for the launch and evaluation contract.

This document consolidates the referenced **Mesh Training Strategy** into an
implementation roadmap. It distinguishes the first-stage deliverables from later
training and physics work. See [PROGRESS.md](PROGRESS.md) for implementation status,
[README.md](README.md) for the current API, and
[docs/validation.md](docs/validation.md) for measured validation.

**Current priority:** prove the method with the original width-16 CNN on richer
geometry/material distributions and better-qualified fine-grid references. Larger
CNNs are deferred until this proof of concept passes physical validation. The
v6 corpus now has 2,000 accepted references. The next dataset addition is 1,000
sparse v7 references, covering one dielectric, close dielectric pairs, one PEC
rectangle, and PEC/dielectric pairs across scales and locations. See
[the sparse supplement contract](docs/sparse_supplement_v7.md) for exact quotas,
feature/gap limits, convergence policy, split protection and combined training.
Later teacher preparation and physics search must include all eight dense/sparse
families and report their physical errors separately. Preserve the completed v6
search targets; its distillation projection timeout needs resolution before a
subsequent distillation run.

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

**Current contract (v6):** PEC is restricted to axis-aligned rectangles and thin
wires, with four and three mandatory mesh lines respectively. PEC takes
precedence over dielectrics. Shapes are defined on the 1/64 PEC coordinate
lattice; probes use 1/128. Material averaging remains enabled. The new ten-scene
pilot uses a separate v6 manifest and retains all outcomes. See
[the v6 campaign contract](docs/reference_campaign_v6.md). Earlier v5 descriptions
and diagnostic subsections below document the development history; this contract
supersedes their mixed-shape PEC distribution.


### 11.1 Scope and preserved baseline

Use `ResUNet(width=16)` (128,578 parameters with ten raw channels), existing FiLM
blocks, ten raw-physics channels, axis-density output and constrained mesher. The
tenth channel is `sigma_h/(2*pi*f_max*MU0)` and remains in the model contract,
but is identically zero in this campaign because magnetic loss is inactive. Start a
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

The v5 generator and campaign implementation are the source of truth for this
section: [generate_v5.py](src/fdtdmesh/data/generate_v5.py) and
[campaign.py](src/fdtdmesh/data/campaign.py). They use four deterministic topology
strata (separated, contact, overlap and nested), with quotas enforced per split.
The implemented distribution requests 8–16 primitives, with realized counts
recorded in the manifest; every returned primitive must retain sufficient visible area; rejected layouts
are resampled, so larger requested counts can be underrepresented.

| Topology stratum | Campaign treatment |
|---|---|
| Separated | Independent quota; near/far separation and resolved gaps |
| Contact | Independent quota; curated shared-edge rectangle pair |
| Overlap | Independent quota; curated intersecting rectangle pair plus randomized geometry |
| Nested | Independent quota; curated nested rectangle pair plus randomized geometry |

Use 8–16 primitives in the ordinary distribution and record the realized count.
Vary translation, orientation, aspect ratio, size ratios and occupied fraction. This campaign uses
circles, rectangles, triangles and convex polygons, with random whole-scene
quarter-turns/reflections and continuously rotated polygons. Tangency, general
junctions, broader contact families, concave polygons and PEC-line topology strata
are deferred; only the curated rectangle pair provides contact/overlap/nesting
structure. Vary global domain aspect/electrical size and local feature scales
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

Keep the PML collar vacuum and keep objects, sources and probes out of it. Require
at least one finite PEC body in every scene. Thin axis-aligned PEC lines are
supported as an optional 0–2 per scene; their endpoints and line positions are
mandatory anchors. Place
probes using the final material map instead of rejecting the whole bounding box
of an overlapping assembly. Maintain explicit source/probe/interface clearance
for the supported interpolation stencil. Start with vacuum probes and one source;
embedded-material probes are a later interpolation-validation extension.

Keep 128x128 CNN inputs initially. Require every finite primitive's normalized minimum
span to be at least 0.125, every visible core radius to be at least 3 pixels at
128, and every visible fraction to be at least 0.25. Reject hidden or tiny
components and thin necks after composition. Zero-width PEC lines are exempt from
finite-body area/core checks precisely because they are anchored, but must span at
least 0.125 of the domain and render as at least 8 pixels at 128. Test composite features after overlap,
not only the original primitive extents. Exact zero-width contacts and anchored PEC
lines need explicit topology/anchor tests and their own convergence statistics.
If useful geometry cannot be represented, test 256x256 inputs with the *same*
width-16 CNN as a separate deferred ablation; version raster/target metadata and retrain.
Do not expect finer FDTD labels to recover information absent from CNN inputs.

### 11.3 Electric parameters and inactive magnetic-loss channel

Use positive, isotropic, nondispersive constitutive parameters within the current
solver contract. The ranges below define synthetic numerical experiments; they
are not broadband constitutive models for particular manufactured materials.

| Quantity | Implemented ordinary distribution | Coverage requirement |
|---|---|---|
| `epsilon_r` | Independently stratified log draws over 1–30 | Cover the full ordinary range without a compulsory high-epsilon tail |
| `mu_r` | Fixed at 1 | Magnetic material variation is deferred |
| `sigma_e` | Sampled from carrier loss ratio `1e-4–1`; 25% exactly zero | Store physical conductivity in S/m and audit realized loss ratios |
| `sigma_h` | Fixed at 0 (`magnetic_loss=False`) | Magnetic damping capability remains tested but is inactive in v5 |
| PEC bodies | Required, with 10% categorical body sampling where applicable | Every scene contains at least one finite PEC body |
| PEC lines | Optional 0–2 axis-aligned zero-width lines | Endpoints and line positions are mandatory anchors |

For nonzero electric conductivity, stratify the dimensionless ratio
`r_e = sigma_e / (2*pi*f_ref*EPS0*epsilon_r)` over `1e-4–1`, using the recorded
source carrier as `f_ref`, then store the resulting physical `sigma_e` in S/m.
Magnetic-loss solver capability is retained for separate tests, but `mu_r=1` and
`sigma_h=0` throughout this campaign. The SI H-conductivity convention is
documented by the [Meep magnetic-loss reference](https://meep.readthedocs.io/en/latest/Materials/)
for conceptual context; it is not used to generate v5 campaign materials.
Record the physical conductivity range and realized ratio distribution. Include
matched controls with one parameter varied at a time and independently crossed
epsilon/loss bins. Magnetic variation is deferred.
Any calibrated physical conductivity bounds and resampling bias must be recorded.

Audit refractive-index and impedance proxies, interface contrasts and electric-loss
attenuation over the excitation band. Lossless local wavelength alone is not a
sufficient resolution estimate for highly conductive objects. Very thin conductive
layers may need extra refinement or an explicit resource rejection.

Do not retain the previous material-OOD range 36–120. Initially evaluate unseen
electric joint combinations within the ordinary epsilon and electric-loss ranges.
Magnetic-material/loss variation is a separately versioned follow-up.

The v5 raw contract has ten channels, including `mu_r`, normalized electric
conductivity, and `sigma_h/(2*pi*f_max*MU0)`. Checkpoints use format 3 and reject
old format-2 checkpoints. Native magnetic damping is implemented and under test,
but is inactive in this campaign and not production validated. Keep the channel
order and normalization in the checkpoint metadata.

### 11.4 Source, scale and distribution coverage

Vary source/probe coordinates, assembly orientation relative to propagation,
source-to-object and receiver-to-object distances, domain aspect, occupied area,
electrical size and bandwidth. Retain the existing physical point-current
normalization. Global amplitude variation alone in this linear solver is not a
meaningful new meshing task under normalized errors.

With `anchor_probes=true` in `SceneSpec`, every point-source/port and receiver
`x`/`y` coordinate is a hard anchor. Legacy scenes omit this field and retain the
default `false` for hash compatibility. v5 source, port, receiver and PEC-line
positions use the normalized 1/128 or 1/64 lattice, so they remain representable
through the 128-grid reference doublings. Validate these anchors before meshing
and retain them in the scene manifest.

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

Implemented campaign reference policy:

| Setting | Next policy |
|---|---|
| Precision | Float64 reference solver |
| Uniform levels | 128, 256, 512, 1024, 2048, 4096; 4096 is this campaign's hard spatial limit |
| Minimum accepted level | 1024, plus geometry/wavelength/attenuation resolution checks |
| Spatial acceptance | Both per-receiver waveform and complex-spectrum relative errors <=2%, for two consecutive refinements |
| Initial duration | At least 48 band-reference cycles and pulse end + 8 conservative material-transit times, including both epsilon and mu |
| Tail acceptance | Pulse ended before the final 20% window; receiver RMS/peak <=1% |
| Duration extensions | Up to 6 doublings from the new base duration; restart spatial sequence on every extension |
| Work ceiling | 20,000,000,000,000 cell updates per attempted grid/window; log cumulative work per scene |
| Field/history estimates | 6 GB field estimate and 1 GB receiver-history estimate per job |
| Resolution / sampling | 16 cells/wavelength and 4 cells/attenuation length; at least 16 time samples/period, cap 262145 timepoints and 32769 frequencies |
| Hard limits | 6 hours per scene; separately, 20,000 candidate indices per split/lane halt an unfilled campaign |

Initial duration uses a conservative bound based on material propagation, e.g.
`sqrt(max(epsilon_r)*max(mu_r))*domain_diagonal/C0`, plus the pulse duration.
It is a starting window, not a prediction of resonance lifetime. Material loss,
contact, PEC cavities and multiple scattering require observed ring-down checks.
Keep units/frequency conventions explicit in the manifest; avoid applying a second
unrecorded duration multiplier to the already enlarged base window.

The 6 GB field and 1 GB history limits are explicit campaign bounds, not a claim
that every reference level fits. Coefficient construction, temporary arrays, solver
copies, histories and postprocessing also consume memory. Record resource-limited
outcomes and retain the attempted level/window history.

History/observation and minimum-level controls are exposed in the reference CLI;
the campaign identity records all additional resolution, quota and wall-time
settings. Per-level progress is persisted. Cumulative work is available by summing
recorded level diagnostics; it is not a separate enforced scene-work cap.
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

1. **Generator and material correctness.** Use the implemented v5 configuration/schema/CLI,
   contact/overlap construction, visible-feature checks, material distributions,
   stable split streams and anchor placement. Test priority, exact contacts,
   deterministic hashes, leakage, probe placement and rendered distribution
   coverage. Produce a gallery and material/topology statistics before a sweep.
2. **Numerical qualification.** Use a separate roughly 160-scene engineering pilot
   balanced over the four topology strata plus independent analytical fixtures.
   It is a development set, not an untouched test set. Profile 2048/4096 and long
   windows, qualify memory/sampling limits, and inspect convergence curves and
   rejected cases. Freeze the acceptance/configuration contract before production.
3. **Versioned proof-of-concept corpus.** Target 1,024 train, 128 validation and
   128 test scenes (1,280 accepted total); continue replacement attempts until each
   split/topology quota is met. These are targets, not generated results. Run:
   `python -m fdtdmesh.data.campaign --output artifacts/reference_v5 --gpus 0 1 2 3 --train 1024 --validation 128 --test 128`.
   Resume only with the campaign's exact config binding; changing generator,
   reference, source, native-binary or quota identity requires a new output root.
   Publish `accepted_manifest.json` and `complete.json` only after all quotas and
   accepted artifacts pass finalization. Every failed, rejected and raw reference
   remains retained; unexpected solver errors halt the worker. Logs include level
   progress. `scripts/qualify_v5.py` creates a development gallery/audit. Extend
   training toward 2,048 only after accepted coverage is measured, preserving
   holdouts exactly.
4. **Width-16 training from scratch.** Generate restart-safe heuristic targets,
   retaining infeasibilities, for budgets 48/64/96/128. Sample feasible pairs uniformly and normalize
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

This revision implements the generator, retains magnetic-loss solver capability,
and defines resolution checks and the quota-driven reference campaign. Magnetic
loss is inactive in v5 scenes. New teacher targets and CNN training remain
deferred until reference generation finishes. See the campaign run record in
[PROGRESS.md](PROGRESS.md).

### Sampled material mapping qualification

Implemented Ez dual-cell filling-fraction sampling for epsilon_r and sigma_e,
with nonuniform physical cells, overlap priority, adaptive 8–32 midpoint samples,
and conservative exclusion of PEC bodies/boundaries and anchored lines. Magnetic
parameters remain fixed. New quota campaigns select this method; the running
ten-scene point-sampled pilot stays a separate baseline. Method and quadrature
settings are part of the reference identity.

Qualification shows about 30× smaller resonance-frequency error in controlled
lossless/lossy dielectric cavities at 512. On pilot scene 000000, 512→1024 waveform
and spectrum differences improve only marginally and remain above 2%. Higher
sampling density has a much smaller effect than the remaining spatial error.
Do not assume dielectric averaging fixes PEC staircasing, corners, or campaign
acceptance. A conformal PEC study is a separate next step, not implemented here.
See [the method and measured results](docs/sampled_material_averaging.md).

### PEC-face alignment experiment

An opt-in `scene.add_pec_anchors()` helper now anchors axis-aligned PEC faces,
line endpoints and conservative ordinary cutouts before remeshing. A feature
mode also supplies polygon vertices/circle extrema without claiming conformal
curved boundaries. The running pilot and default reference mesh policy remain
unchanged. A separately identified nonuniform reference policy is needed before
adopting this in bulk generation.

A controlled flat-wall cavity improved 44× at 512. On pilot scene 000000, adding
four PEC-face coordinates and using nested graded grids reduced the 512→1024
waveform/spectrum differences to 2.542%/2.495%, from 3.395%/3.529% with averaged
materials on uniform grids. Both still exceed 2%; timestep count increased 26.7%.
The fuller feature-anchor mesh timed out, rather than proving infeasibility.
See [PEC anchor experiment](docs/pec_anchor_experiment.md) for the measured scope.

### Rectangle-only PEC follow-up

The same-scene diagnostic with only the PEC rectangle retained meets convergence
criteria at 1024 for both uniform and PEC-face-anchored grids, with dielectric
averaging enabled. Anchored waveform/spectrum differences are 0.02835%/0.03422%,
about ten times smaller than uniform-grid differences. The five dielectric
objects and all probes remain unchanged. This supports testing axis-aligned
rectangle-only PEC on a broader pilot before changing the bulk generator.
Keep exposed PEC boundaries orthogonal: later curved dielectric cutouts can
otherwise reintroduce non-rectangular PEC interfaces. See
[the full diagnostic](docs/rectangle_only_pec_experiment.md).

### Active reference campaign: 2,000 converged scenes (2026-09-19)

Following the four-scene v6 pilot (4/4 converged at 1024), the authorized campaign
target is now 1600 training + 200 validation + 200 IID test references. This
supersedes the earlier 1024/128/128 quantity; geometry, PEC/probe anchors, sampled
material averaging, strict convergence gates and resource hard limits are
unchanged. The active spatial sequence is 128/256/512/1024/2048. Failure to
converge by 2048 is terminal for that candidate and triggers replacement; only
an unsettled time tail can extend simulation duration. Detached execution uses
GPUs 0–3 and writes to
`artifacts/reference_v6_2000/`. Rejected candidates do not count toward quotas.
Generation completed with all 2,000 accepted references on 2026-09-20.
The user authorized training; the initial teacher-pretraining workflow is now
launched. See [v6 training](docs/training_v6.md) for the source-bound target cache,
balanced budget sampling, width-16 optimizer configuration and subsequent
physical-validation requirements.

The 2048 ceiling was applied as a recorded policy migration after 552 decisions.
Seventeen 4096-only accepted cases were reclassified as nonconverged, and four
interrupted attempts were archived before restart. The migration record and old
identity are preserved in `policy_migration_max_2048/` beneath the campaign root.

### Baseline naming and PEC discretization (2026-09-20)

Use actual uniform spacing without PEC/probe anchors for the `uniform` candidate,
snapping PEC rectangle faces and wire coordinates/endpoints to their nearest
mesh lines. Rename the former anchored constant-density baseline `quasi_uniform`.
Keep heuristic/CNN meshes anchored, and retain exact reference geometry. The
true uniform candidate supplies a benchmark and work normalization, not an
anchored training target. See [baseline conventions](docs/uniform_baselines.md).
