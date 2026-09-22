# Simple-first scattering curriculum and joint loss

The user asked us to follow the advice in
[CNN Mesh Prediction Using RCS Loss](chatgpt-conversation://6ab1f808-3d48-83eb-a625-d0261719186c).
The discussion was read on 22 September 2026. Its recommendations are design input;
solver claims are validated independently in this project.

## Sequence

1. **One PEC circular cylinder.** This is the first training family. Use analytic
   complex far fields. Vary k0*a, location relative to Yee cells, incidence angle,
   frequency, and budget. Start with a proposed 32–64 base geometries and several
   candidate meshes per condition; measure diversity and cost before scaling up.
2. **One dielectric cylinder.** Use analytic references again. Begin with moderate
   lossless contrast, then include epsilon_r up to 30 and conductivity. Keep mu_r=1
   and sigma_h=0. Add resonant/high-contrast cases after time-settling qualification.
3. **Two and multiple cylinders.** Introduce gap distance and size contrast gradually.
   Qualify mixed PEC/dielectric coupling before generating those combinations.
4. **Rectangles and corners, then thin screens.** Add only after PEC topology and
   boundary handling are qualified. Keep family-specific accuracy/failure reports.
5. **More varied shapes and complex arrangements.** Use them for later expansion
   and held-out-shape tests, not as the initial bulk training distribution.

For single cylinders, analytic series evaluation replaces expensive fine-FDTD
reference generation. Check series truncation and phase conventions. Candidate
FDTD runs still require source/PML/NF2FF/time qualification; an analytic target
does not make the candidate numerical solution converged.

Do not start a large campaign or CNN fit until low-budget candidate searches show
useful accuracy-versus-cost differences. The existing fixed-focus examples are
solver checks; their performance does not establish an optimal meshing policy.

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

- Enforce a computational cap based initially on Nx*Ny*Nt, including PML, alongside
  requested Nx/Ny. Report wall time and enlargement overhead. Equal cell count
  alone is not equal compute. Use the same required physical duration for every
  mesh and reject unsettled or invalid candidates before ranking.
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

No new training or data-generation campaign is launched by this planning update.
