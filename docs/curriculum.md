# Simple-first scattering curriculum and joint loss

The user asked us to follow the advice in
[CNN Mesh Prediction Using RCS Loss](chatgpt-conversation://6ab1f808-3d48-83eb-a625-d0261719186c).
The discussion was read on 22 September 2026. Its recommendations are design input;
solver claims are validated independently in this project.

## Product target

The learned mesher is intended for sparse scattering scenes. Sparse does not mean
only one object: a scene may have several dielectric or PEC objects and close gaps,
but the geometry and required fine-mesh regions remain localized within the larger
domain. Because the output is a tensor-product grid, record sparsity in projected
x and y as well as occupied area. Objects scattered across the full domain can use
most axis lines even when their total area is small, eliminating the main advantage
over a uniform grid. Dense arrangements are held-out stress tests, not the target
distribution.

## Sequence

1. **Single-object numerical controls.** PEC and dielectric cylinders use analytic
   complex far fields and vary k0*a, subcell location, incidence, frequency, and
   budget. The isolated PEC candidate search had no low-budget nonuniform headroom,
   so PEC remains a solver/control family rather than the first bulk training set.
2. **One dielectric cylinder.** This is the first learned family. Begin with moderate
   lossless contrast, then include epsilon_r up to 30 and conductivity. Keep mu_r=1
   and sigma_h=0. Add resonant/high-contrast cases after time-settling qualification.
3. **Sparse localized pairs.** First qualify two dielectric cylinders with log-spaced
   gaps, radius ratio, material contrast, cluster scale, translation, and incidence.
   Then add circle--rectangle and rectangle--rectangle topology together with
   dielectric--PEC and PEC--PEC material pairs. Keep topology/material strata
   separate in every promotion report. The pair remains localized; close gaps create
   local difficulty without filling the domain.
4. **Sparse 3–4 object and PEC/mixed clusters.** Extend the qualified pair primitives
   to separated PEC/dielectric objects, including circular PEC as well as PEC
   rectangles, corners, and later thin screens. Keep contacting/overlapping
   interfaces excluded until defined. The 48-scene pilot in
   `configs/sparse_cluster_pilot_32.json` has separate three- and four-object
   PEC-circle families in train, validation, and test splits. Circular PEC is part
   of the final multi-object training and evaluation target, even though its
   isolated two-circle gap pilot favored uniform meshes. Preserve the uniform
   candidate when it gives the best label; do not remove circular PEC scenes
   merely because a nonuniform policy does not win on them.
5. **Dense and complex arrangements as stress tests.** Use domain-spanning scenes,
   intersections, and resonance-heavy assemblies to measure failure modes. They do
   not dominate training and cannot compensate for a sparse-family gate failure.

Initial sparse-pilot strata target roughly 0.5–12% occupied area and at most 45%
projected feature support on either axis. Save the measured occupancy, cluster
envelope, projected support, object count, PEC fraction, and minimum gap. Translate
the complete cluster throughout the domain while keeping related variants in one
data split. Combined batches allocate at least 70% sampling weight to sparse
production scenes, at least 20% to analytic single-object controls, and no more
than 10% to dense stress scenes.

For single cylinders, analytic series evaluation replaces expensive fine-FDTD
reference generation. Check series truncation and phase conventions. Candidate
FDTD runs still require source/PML/NF2FF/time qualification; an analytic target
does not make the candidate numerical solution converged.

The circular-PEC gap pilot in `configs/pec_circle_gap_pilot.json` compares exact
32/48/64 meshes for two PEC circles across close, moderate, and wide gaps and
horizontal/vertical orientations. Uniform, interface, and three gap-focus strengths
are scored against converged 192/256 references using the same complex far-field,
RCS, and soft-Nt objective. Preserve the uniform policy in the teacher search:
one held-out PEC-rectangle pair at 48×48 had lower scattering loss on uniform
than on an interface-focused mesh, so finer boundary spacing is not presumed to
help under conformal PEC treatment. Candidate feasibility and per-gap outcomes
decide whether gap-focused labels should enter the multi-object curriculum.

The completed six-scene circular-PEC pilot qualified all 12 references (worst
192/256 joint loss 1.16e-5) and all 90 candidates. Uniform had the best soft-Nt
score in 17 of 18 scene-budget conditions; weak gap focus won only the moderate,
horizontal 32×32 case by 1.056×. Median gap-focus Nt was 1.63–2.04× uniform,
and the pilot failed its declared nonuniform-headroom gate. Keep these circular
PEC pairs as controls rather than forcing gap-refinement labels. The stage-four
multi-object pilot still contains PEC circles with dielectric/PEC neighbors and
will test whether their joint geometry creates useful localized headroom. Track
the three-object one-PEC-circle and four-object two-PEC-circle families separately
in held-out physics reports, with complex far-field/RCS error and uniform-versus-
learned mesh cost at each exact budget and gap stratum. A final multi-object
promotion cannot rely solely on aggregate results that hide circular-PEC failures.

Do not start a large campaign or CNN fit until low-budget candidate searches show
useful accuracy-versus-cost differences. The existing fixed-focus examples are
solver checks; their performance does not establish an optimal meshing policy.

That gate now passes for moderate-contrast dielectric cylinders. The first generated
pool, `simple_dk_3ea40e8117434e5c`, contains 32 geometries in eight four-variant
lineages. Lineages are assigned wholly to train/validation/test (24/4/4 geometries),
and held-out splits use disjoint incidence angles and size regimes. Expansion over
their declared illuminations and 32/48/64/96 budgets creates 352 conditions before
mesh candidates. See `configs/simple_dielectric_pool.json`. Candidate physics runs
and label selection are complete for this first pool, but its frozen training gate
fails because all validation labels select uniform. It remains a measured search
dataset, not CNN training data. The next revision independently varies radius,
permittivity, and conductivity and searches intermediate nonuniform resolutions
under each uniform compute cap.

The implemented replacement is `simple_factorial_69168792914bfe03`. It contains
148 geometries in 74 two-variant lineages and expands to 1,552 conditions. The
low/moderate training tier independently varies four permittivities, three radii,
and two positive conductivities, with separate epsilon_r={2,4} lossless controls.
The settling-qualified high-contrast tier reaches epsilon_r=30 with loss tangent
0.10--0.20. Validation and test use disjoint material levels and incidence angles.

The six exact-axis policies preserve every requested Nx and Ny. They are ranked by
`joint_scattering_loss * (Nt/Nt_uniform)^0.1`; strict matched-update comparisons
remain a separate diagnostic. The 312-case pilot passed the label-diversity gate,
and the full 9,312-case campaign is restartable with 70/140/560 ns duration retries.

The current 96-scene sparse-pair campaign has 72 train, 12 validation, and 12
test geometries, balanced across dielectric pairs, mixed dielectric/PEC pairs,
and PEC rectangles. After its convergence and headroom gate passes, append its
288 exact-budget conditions to the sparse dataset. Warm-start each of the nine
128/256/384-resolution, 16/24/32-width CNNs from its matching verified checkpoint
at learning rate 1e-4, then evaluate complex far-field and RCS error on all 90
old-plus-new held-out sparse-pair conditions. Select using validation physics and
inspect PEC and low-budget strata separately; test remains a final check. The
48-scene multi-object pilot, including circular PEC, is queued behind the pair
campaign and will supply the next distinct sparse-cluster family if qualified.

## Targets and loss

Save F(theta,f)=A/Eincident as complex128 or paired real/imaginary arrays. Save the
incident spectrum, phase origins, observation angles, frequencies, and Fourier
convention. Derive the 2D RCS/scattering width as sigma=2*pi*abs(F)^2 in metres.

Use both terms:

    L_FF = mean_f [ mean_theta |F_pred-F_ref|^2
                   / (mean_theta |F_ref|^2 + F_floor^2) ]

    sigma_floor(f) = max(1e-4 * max_theta sigma_ref(f), 2*pi*F_floor^2)
    delta_dB = 10*log10((sigma_pred+sigma_floor)/(sigma_ref+sigma_floor))
    L_RCS = mean((delta_dB/(20/ln(10)))^2)
    L_accuracy = L_FF + 0.25*L_RCS

The weights and -40 dB reference-power floor are **initial pilot settings**.
The dB scaling is explicit so 0.25 is not mistakenly applied to raw dB-squared
error. Calibrate the amplitude floor to physical scale/numerical noise and review
component distributions and mesh rankings before freezing the training loss.

`scattermesh.metrics.scattering_loss` implements this NumPy score for candidate
ranking. It is not a differentiable CNN training implementation. Keep both raw
components in records; adjust weights using saved complex spectra without new
FDTD runs. Never fit away a global phase offset or use wrapped phase MSE as the
sole phase loss. Phase-only diagnostics exclude scattering nulls.

## Budgets, labels, and splits

- Preserve the requested Nx and Ny exactly. Rank with a configurable soft Nt cost,
  and report raw Nt, cell updates, and wall time separately. Equal cell count alone
  is not equal compute. Reject unsettled or invalid candidates before ranking.
- Generate multiple feasible density/mesh candidates per geometry, illumination,
  and budget. Rank using joint accuracy and record the Pareto set. Preserve several
  good meshes rather than declaring one exact coordinate sequence to be truth.
- A small CNN/residual U-Net predicts refinement importance; a deterministic stage
  projects it to legal x/y densities. Start with ranking or set-valued distillation;
  a direct differentiable solver/mesh path is later work.
- Group geometry lineages and their angles, translations, scales, and candidate
  meshes into one split. Explicitly hold out electrical-size intervals and angle/
  position regimes in the cylinder stage. Later hold out shapes and topologies.
- Compare uniform, wavelength-based, interface-based, solution/error-adaptive, and
  learned meshes with error-versus-update/runtime curves. Report simple, sparse,
  and complex families separately as they become available.
- Make the primary promotion decision from the sparse strata at low exact budgets.
  Report results by occupied area, projected support, gap, object count, PEC fraction,
  and cluster location. Dense stress performance is diagnostic and never masks a
  sparse-family regression through an aggregate average.
