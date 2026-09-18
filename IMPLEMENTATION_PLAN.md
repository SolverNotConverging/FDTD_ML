# Learned FDTD meshing implementation plan

Updated: 2026-09-18.

This document consolidates the referenced **Mesh Training Strategy** into an
implementation roadmap. It distinguishes the first-stage deliverables from later
training and physics work. See [PROGRESS.md](PROGRESS.md) for implementation status,
[README.md](README.md) for the current API, and
[docs/validation.md](docs/validation.md) for measured validation.

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
   shortcuts. The mixture is an initial design choice, pending application statistics.
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
