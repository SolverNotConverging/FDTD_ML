# Plane-wave scattering and low-budget learned meshing

Status: new project, 22 September 2026. The receiver-waveform CNN project is retired.
The CPU scattering solver and experimental conformal PEC/cell-enlargement modes
are implemented; production reference generation and CNN training are not started.
See [progress](PROGRESS.md) for measured validation.

## Objective and decision gates

Learn where a limited number of nonuniform Yee cells most improve scattering
accuracy. The primary experiment is at low budgets, with the same continuous scene,
incident wave, physical domain, and simulation stopping criteria for every mesh.
**The physics loss includes both source-normalized complex far-field error and
RCS/scattering-width error. Complex targets retain absolute phase; both terms are
stored and reported separately.** See the [simple-first curriculum](docs/curriculum.md),
updated from the user's referenced discussion.
Count PML cells and the time-step penalty from small cells in the computational cost.
Do not infer success from teacher imitation loss, early stopping, or visual mesh density.

Before another large campaign, demonstrate that optimizing mesh placement actually
improves scattering accuracy on a small held-out suite at matched cell and update
budgets. If it does not, investigate solver, geometry representation, and mesh
parameterization before allocating more training compute.

## Archived project

- Source snapshot: branch `codex/archive-receiver-cnn-20260922`, commit `5878ddd`.
- Physical archive: `archive/receiver_cnn_2026-09-22/`, including code, documentation,
  reference manifests, checkpoints, plots, and compiled outputs.
- Archive integrity: `ARCHIVE.json`, `INVENTORY.json`, `INVENTORY_SUMMARY.json` there.
- Active branch: `codex/plane-wave-scattering`; independent package `scattermesh`.
- The root `artifacts` symlink preserves old result links. Shared Python environments
  remain available. Old datasets and checkpoints are historical, not new targets.

## Numerical contract

1. Start in 2D TMz: fields Ez, Hx, Hy, SI units, infinite extent along z. Report
   **2D scattering width in metres**, not 3D RCS in square metres. TEz/3D require
   separate solvers and validation; TMz cannot establish every gap/polarization effect.
2. Vacuum background; fixed mu_r=1 and sigma_h=0. Dielectrics have epsilon_r in
   [1, 30] and nonnegative sigma_e. Dataset conductivity ranges will be specified
   through both S/m and loss tangent at a declared reference frequency.
3. Cartesian tensor-product nonuniform Yee mesh with exact requested Nx, Ny.
   Configurable adjacent-cell ratio caps 2 and 3; 1.4 is an ablation, not the default.
   Positive widths, conservative CFL, finite fields, and valid geometry remain hard
   constraints. A loss cannot rescue unstable updates or an unrepresented PEC wire.
4. No point TX/RX anchors. DFT contours select existing mesh nodes. Dielectrics
   and the experimental conformal PEC modes require no object-boundary anchors.
   Any future fallback alignment must be explicit and counted in the budget.
5. Sample material filling fractions over actual Ez dual-cell areas. For this TMz
   polarization Ez is tangential to extruded dielectric interfaces, so arithmetic
   epsilon/sigma averaging is used. Keep sampling convergence separate from spatial
   and temporal convergence. Material overlaps follow explicit last-object priority.
6. Experimental PEC cut edges now preserve the actual circle/rectangle intersection
   and impose zero total tangential E there. No PEC material averaging or protective
   anchor triplets/quartets. Plain conformal mode respects its cut-cell CFL bound.
   An optional `enlarged` mode couples tiny cut regions to neighboring field regions
   through an energy-consistent Galerkin projection and checks the reduced operator's
   CFL bound. Do not obtain speed by silently clipping cut fractions or shifting PEC.
   Single PEC circles are the first training family, with analytic complex-field
   references. Rectangles/corners follow after cylinder stages. Zero-thickness screens, split edges,
   unresolved gaps, and mixed dielectric/PEC coupling remain explicit next work.
   A 2D thin segment represents an extruded screen, not a finite 3D wire antenna.

## Source design and openEMS review

The current reference implementation evolves scattered fields everywhere and drives
material contrast using a broadband analytic plane wave. For fixed mu_r=1:

    epsilon * dEs/dt + sigma * Es = curl(Hs)
                                      - (epsilon - epsilon0) * dEi/dt - sigma * Ei
    mu0 * dHs/dt = -curl(Es)

The incident signal is a Gaussian-modulated carrier evaluated at
`t - dot(direction, position - phase_origin)/c0`. This preserves incidence angle
across the pulse bandwidth. Conductivity uses a trapezoidal update; incident E is
evaluated at the same electric time levels. CPML absorbs scattered fields.
No plane-wave injection box is needed for this formulation. Vacuum produces zero
scattering by construction, so that test alone does not validate the source.

openEMS instead implements TFSF. We inspected its source at commit
`8085f3129d4b41835e6e96365cb75218d60ef029`; see
[review and exact links](docs/openems_review.md). Its optional dispersion correction
uses one frequency and average cell widths. Our inference is that this is useful
guidance, but cannot certify arbitrary strong grading over a broadband pulse.

Keep a TFSF implementation as a comparison milestone: snap its rectangle to existing
lines, evaluate E/H on their own staggered positions and times, apply local curl
coefficients and correct signs, and test empty-box leakage before object scattering.
Choose the production source using analytic and cross-solver evidence. Do not
introduce point-source anchors or silently transplant a single-frequency phase-speed
correction into the broadband contrast-source formulation.

## Milestones and completion criteria

### M0 — archive and CPU numerical reference (implemented foundation)

- Independent NumPy float64 solver with nonuniform curl metrics, sampled dielectrics,
  lossy contrast forcing, CPML, oblique plane waves, and bounded execution.
- Streaming complex surface DFT with E/H half-time-step correction and physical
  quadrature weights; no full time-history storage.
- Closed-contour 2D NF2FF; complex amplitude, angular scattering width, source-spectrum
  normalization floor, field-tail and cost diagnostics.
- Persist complex128 normalized far fields and incident spectra with frequency,
  observation angle, Fourier sign, incident phase origin, and far-field origin.
  Test absolute complex cylinder phase and subcell-translation phase factors.
- Tests against outgoing Hankel waves and analytic lossless/lossy cylinder scattering.
- Reproducible `scripts/qualify_scattering.py`, report and scientific plot.
- Experimental off-grid PEC rectangles/circles with staircase, conformal, and
  enlarged comparisons in `scripts/pilot_conformal_pec.py`. Test exact geometry,
  positive projected energy/stiffness, tiny cut fractions, and analytic scattering.

This is an executable starting point, not a qualified reference-data service.

### M1 — broader numerical qualification and PEC

- Extend the implemented PEC rectangle/cylinder prototype to split-edge thin
  segments and mixed dielectric/PEC scenes; preserve total-field cancellation.
- Qualify enlargement across location/scale/angle, strong grading, close gaps,
  cavities, resonant structures, and late-time energy. A local aggregation can
  sacrifice boundary accuracy; measure its error and cost against unreduced cuts.
  Retain automatic CFL reduction when aggregation cannot safely recover the
  background step. Do not infer a universal no-penalty method from isolated bodies.
- Test displaced/scaled cylinders, epsilon_r up to 30, conductive dielectrics,
  resonances, negative and oblique incidence directions, and abrupt ratios near 2/3.
- Vary mesh, material quadrature, dt/CFL factor, runtime, PML thickness/resolution,
  and NF2FF contour independently. Check complex far-field amplitude and width.
- Add an openEMS-style TFSF comparison with empty-domain leakage, transmitted-wave
  amplitude/phase, and interface tests. Add an external openEMS benchmark with
  matched polarization/dimensional interpretation; do not compare 2D width to 3D RCS.
- Initial engineering targets: analytic angular L2 complex-field error <1% on resolved
  benchmarks; duration/PML/contour/quadrature variations each <0.5%. These are
  proposed gates, to be calibrated across resonant and weak-scattering cases.
  Also report magnitude, width, and phase errors. A width-only pass is insufficient.
- Report angular nulls with absolute and floor-normalized errors, not unstable
  pointwise relative percentages. Empty/no-contrast cases require absolute metrics.

### M2 — CUDA solver and observables

- Port validated update equations and coefficient construction without importing
  legacy experiment policy. Keep fields, CPML state, source evaluation, and DFT on GPU.
- Implement fused/batched surface DFT kernels; use phase recurrence with bounded
  drift or direct phases validated against CPU. Preserve half-step H timing.
- Port PEC intersection metadata, cut-edge updates, and conservative aggregation
  transfers. Avoid race conditions in shared-master reductions; validate geometry
  and field constraints after each GPU update on small tests.
- Implement NF2FF reductions after time stepping. Start with accurate accumulation;
  quantify float32 versus float64 and reduction errors before reducing precision.
- Test CPU/GPU equivalence on uniform, graded, lossy, PEC, and oblique cases;
  benchmark memory, wall time, cell updates, and DFT overhead separately.
- Batch compatible scenes if beneficial; no host-device field transfer each step.

### M3 — prove low-budget mesh headroom before training

- Pilot 32/48/64 cells per axis and 96/128 controls, subject to actual feasibility.
  These are proposed scattering budgets, not continuation of the retired campaign.
- Compare true uniform, simple feature-based nonuniform, randomized density, and
  a small direct search over density parameters; later add CNN meshes.
- Compare both equal total Nx*Ny and equal total Nx*Ny*Nt. Record PML fraction,
  movable lines, smallest cells, grading, geometry displacement, and runtime.
- Allow independent Nx/Ny and unseen budget pairs. Vary grading caps 1.4/2/3.
- Separate meshing error from source, PML, material-sampling, and NF2FF error.
  A coarse mesh with a shifted PEC boundary must report the geometry error.
- Require a measurable nonuniform advantage on representative low-budget cases.
  Current focused-cylinder meshes are solver tests, not an optimized teacher.

### M4 — new scene schema and converged references

Start in controlled stages rather than immediately mixing every geometry family:

1. Single PEC circular cylinders: vary electrical size, subcell location, incidence,
   frequency band, and compute budget; analytic complex references.
2. Single dielectric cylinders: start with moderate lossless contrast, then extend
   epsilon_r to 30 and add conductivity; analytic complex references.
3. Two/multiple cylinders: introduce controlled gaps and size/material contrasts.
4. Rectangles/corners and thin screens once their solver treatment is qualified.
5. Broader shapes and complex assemblies as generalization/stress tests.

Propose a first pool of only 32–64 base PEC-cylinder geometries for candidate-mesh
experiments, before any large data campaign. Final counts depend on acceptance,
timing, and evidence of mesh headroom. Follow [the staged curriculum](docs/curriculum.md).
Analytic references still need series-order checks and consistent phase conventions;
an over-refined FDTD result is not the default truth for single cylinders.

As stages expand, keep three separately tagged families and report their results separately:

| Family | Initial contents | Controlled difficulty |
|---|---|---|
| Simple | First PEC circle, then dielectric circle; later other single objects | Scale, location, contrast, loss, incidence |
| Sparse | Two dielectrics, two PEC objects, PEC + dielectric | Log-spaced gap, size ratio, orientation, incidence |
| Complex | Several mixed objects with explicit overlap priority | Occlusion, multiple scattering, intersections, resonances |

- Mix locations/scales deliberately; do not fill every domain or use subpixel
  decorations. Require features to span multiple finest-reference cells. Resolve
  geometry in the model input independently of the candidate solver mesh.
- Include empty/near-vacuum controls without allowing them to dominate training.
- Sample incidence continuously over 0–2pi. Store geometry identity separately
  from illumination identity. Group related geometry, translated/scaled variants,
  and their illuminations into the same train/validation/test split.
- Explicit held-out angle sectors and size/gap ranges test extrapolation. Also
  evaluate unseen angles on held-out geometry to measure ordinary generalization.
- Keep shape/topology holdouts once multiple families exist. For the initial
  cylinder-only experiment use explicit held-out electrical-size intervals and
  position/angle regimes; nearby-radius interpolation is not shape generalization.
- Start with tens of qualified scenes, then hundreds; choose full campaign size
  after timing and diversity review. Do not relabel the old 3,000 references.
- Reference records contain immutable scene/config/code IDs, actual x/y, dt/Nt,
  CPML and quadrature settings, native complex far fields (or explicit real/imag
  arrays), derived angular widths, DFT contour, source spectrum, incident/far-field
  phase origins, Fourier convention, settling histories, comparisons, and wall times.
- Acceptance requires complex-field agreement under independent spatial refinement
  and time extension, plus
  monitored-tail and source-spectrum checks. Compare at least two successive fine
  refinements; agreement of two underresolved meshes is not enough. Escalate mesh,
  runtime, and quadrature within explicit limits; retain failed attempts and skip
  unresolved scenes. No field-tail-only convergence labels.

### M5 — new model and training targets

- Start a fresh small CNN/residual U-Net. Condition on continuous-geometry raster/SDF, material
  channels, gap/feature information, frequency band, sin/cos incidence, Nx/Ny, and
  grading policy. Use an input resolution high enough to see the smallest admitted
  features; test subpixel translations and resolution sensitivity explicitly.
- Predict a spatial importance map projected to positive x/y axis densities, or
  predict the two axis densities directly. A free 2D mask is not a realizable
  tensor-product grid. The deterministic mesher enforces exact cell counts and
  checks actual Nx*Ny*Nt against the compute-budget cap.
  Conformal PEC is the preferred candidate to avoid boundary anchors, conditional
  on M1 qualification. Cell enlargement changes effective degrees of freedom;
  record aggregation counts and error sensitivity. Any fallback alignment is a
  separate auditable projection; reject datasets dominated by fixed lines.
- Sample multiple independent Nx/Ny combinations across the range; hold out
  combinations and intermediate budgets from training. Resample budgets during
  training while keeping reference geometry/physics fixed.
- Optional short heuristic pretraining is initialization. Main supervision is
  measured scattering accuracy and computational cost; the teacher must not
  permanently cap the achievable mesh quality.
- The complex component compares real and imaginary parts directly:

      L_complex(f) = sum_angle w * |F_pred - F_ref|^2
                     / (sum_angle w * |F_ref|^2 + F_floor^2)

  Here F=A/Eincident is the complex far field in sqrt(m), angular weights sum to
  one, and F_floor is calibrated to physical scale/numerical noise. No arbitrary
  global phase rotation or time shift is fitted away. This retains interference
  and propagation-phase information. Plain wrapped-angle MSE is not the primary
  loss, and phase at scattering nulls is excluded from phase-only diagnostics.
- Add a floored logarithmic RCS/scattering-width component:

      delta_dB = 10*log10((width_pred + width_floor)/(width_ref + width_floor))
      L_rcs = mean((delta_dB / (20/ln(10)))^2)
      L_accuracy = lambda_complex*L_complex + lambda_rcs*L_rcs

  Scaling the dB term by 20/ln(10) makes its small-amplitude-error limit comparable
  to squared relative amplitude error. The implemented NumPy scorer starts with
  **provisional** weights 1 and 0.25 and an RCS floor 1e-4 of each frequency's
  reference peak, bounded below by an absolute floor. Calibrate these on training
  pilots, then freeze them for validation/test; store components so reweighting
  does not require rerunning FDTD. These are not tuned CNN loss weights.
- Report weighted phase RMS, complex error, and RCS error separately. Include
  worst-frequency performance and hard mesh-feasibility/settling/compute-budget
  gates. Compare Pareto curves against uniform, wavelength/interface heuristics,
  and classical solution/error-guided refinement. Do not inherit old loss weights.
- Report family losses independently before choosing a combined weighted loss.
  Simple/sparse cases should receive substantial sampling and weight because they
  represent the intended use; complex stress cases must not overwhelm them.
- A forward solver alone does not give training gradients. First implement
  budget-conditioned distillation from physics-evaluated candidate mesh searches;
  then evaluate an adjoint/differentiable path as a distinct validated extension.
- Retain multiple good candidate meshes and their ranking, density maps, and cost.
  Avoid one arbitrary ground-truth mesh per scene. Evaluate ranking/set-valued
  supervision in the first model pilot before direct differentiable FDTD.

### M6 — frozen evaluation and campaign operation

- Freeze final test families, illumination settings, budgets, and source/monitor
  protocol. Report median and worst-tail accuracy, failure rates, runtimes, and
  error-versus-cost curves for all baselines and each family.
- Require useful improvement at low budgets, not just near-equality at 128 cells.
- New restartable campaign runner: explicit CPU/GPU ownership, atomic results,
  retry limits, progress/timing script, and no automatic training on partial or
  unqualified reference outputs.

## Numerical references

- [Scattered-field ADE-FDTD paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2763393/):
  field decomposition and material-contrast source formulation.
- [openEMS source review](docs/openems_review.md): staggered TFSF implementation.
- [Meep near-to-far documentation](https://meep.readthedocs.io/en/master/Python_User_Interface/#near-to-far-field-spectra):
  frequency-domain surface observations and homogeneous exterior requirements.
- [2D cylinder scattering benchmark](https://optics.ansys.com/hc/en-us/articles/360042703373-Mie-scattering-2D):
  analytic scattering validation approach.
- [Conformal PEC and cell enlargement experiment](docs/conformal_pec.md):
  implemented 2D method, mathematical stability check, limitations, and sources.
