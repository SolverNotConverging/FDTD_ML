# Stage 2 validation and numerical conventions

Validated 2026-09-18 on Windows, RTX 4070 Laptop GPU, CUDA 13.3, MSVC 14.51,
Python 3.12 and the locked uv environment. The native extension was rebuilt with
`scripts/build_cuda.ps1`. The full suite passes **78 tests, no skips**.
Historical cavity/convergence evidence is retained in [stage-one validation](validation.md).

## Mandatory mesh projection

All meshes enforce maximum adjacent width ratio 1.4, with optional stricter limits.
The mixed-integer problem jointly chooses anchor indices and coordinates, minimizing
normalized mean absolute displacement from density quantile lines. A continuous
LP then polishes that assignment using exact anchor bounds. Final coordinates are
verified before returning. See [grading policy](anchor_grading.md) for examples,
training loss semantics, tolerances and search limits.

Tests cover exact counts/anchors, fixed collars, scale behavior, deterministic
repeated output, recovery of the old allocation failure, and exhaustive enumeration
of all anchor assignments in a small problem. Timeout is distinguished from proven
infeasibility. Density outside fixed collars does not affect the solution, even
when those values differ by 100 orders of magnitude.

## CPML equations and layout

Vacuum collars use cubic profiles by default. At each actual electric or magnetic
Yee coordinate, define normalized collar depth u in [0,1]:

```text
sigma_max = -(m+1) * log(R0) / (2 * eta0 * thickness)
sigma = sigma_max * u**m
kappa = 1 + (kappa_max-1) * u**m
alpha = alpha_max * (1-u) inside the collar; zero elsewhere
b = exp(-(sigma/kappa + alpha) * dt / epsilon0)
a = sigma * expm1(-(sigma/kappa + alpha)*dt/epsilon0)
    / (sigma*kappa + alpha*kappa**2)
psi_new = b*psi + a*field_difference
corrected_difference = field_difference/kappa + psi_new
```

For zero denominator, a=0. Identity interior profiles are (1/kappa,b,a)=(1,1,0).
Magnetic damping uses the matched electric-equivalent rate, sampled at magnetic
half-cell coordinates. Coefficients already divide by the appropriate primal/dual
width, so convolving raw differences is equivalent to convolving spatial derivatives
on this static mesh. Separate x/y electric auxiliaries combine in Ez, including at
corners; Hx/Hy each have their own auxiliary. PEC masks remain enforced.

The formulation follows [Roden and Gedney's CFS-CPML construction](https://doi.org/10.1002/1098-2760(20001205)27:5%3C334::AID-MOP14%3E3.0.CO;2-A).
Integer optimization uses [SciPy's HiGHS MILP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

All four auxiliaries and profiles are allocated/uploaded before the native timestep
loop, remain device-resident, and reset on each new run. Runtime transfer counters
are zero during stepping; this is accounting plus code inspection, not an independent
Nsight trace. CUDA/NumPy parity tests exercise x-only, y-only and xy CPML in both
precisions on nonuniform meshes with an interior lossy magnetic dielectric.
Repeat-run checks verify state reset. No CPU production fallback was added.

## Independent boundary-error benchmark

The acceptance gates were fixed before evaluation: normalized peak waveform and
waveform L2 errors below 1%, final interior field error below 1%, and finite late
fields below 1% of initial interior Ez norm. PEC controls must exceed 10% peak error.

`benchmarks/validate_cpml.py` launches a localized Gaussian packet with 60 GHz
carrier and 1.5 mm envelope width at angles 0, 30, 45, and 90 degrees. The domain
is 24 x 24 mm with 96 x 96 total cells, including 12 uniform 0.25 mm PML cells on
each side. Parameters are order=3, kappa_max=3, alpha_max=0.05 S/m, R0=1e-8.
The nonuniform case uses axis density [1,1,1.2,1.8,1.8,1.2,1,1].

A second PEC domain extends the exact original mesh outward by 100 cells on every
side and uses the same dt, initial packet, and physical probe locations translated
with the mesh. During the 160 ps comparison, boundary echoes from that enlarged
domain have not returned to the observed region. Nine receivers (center and eight
on a 4 mm ring) measure the entire waveform, including reflected arrivals. The
45-degree case exercises corner propagation. The packet is not claimed to be an
exact continuum solution: both runs share the initial discretization. This test
isolates boundary error, not arbitrary-scene spatial convergence.

Peak error is max|CPML-reference| over time and receivers divided by the peak
reference amplitude; L2 uses the full recorded arrays. Final interior error is
normalized by the initial interior Ez norm. Results:

| Grid | Angle | Peak waveform error | Relative waveform L2 | Final interior error |
|---|---:|---:|---:|---:|
| Uniform | 0 | 7.067e-6 | 2.567e-5 | 3.351e-7 |
| Uniform | 30 | 6.241e-6 | 1.784e-5 | 3.302e-7 |
| Uniform | 45 | 6.185e-6 | 1.539e-5 | 3.292e-7 |
| Uniform | 90 | 7.067e-6 | 2.567e-5 | 3.351e-7 |
| Nonuniform | 0 | 5.248e-6 | 1.993e-5 | 1.784e-6 |
| Nonuniform | 30 | 5.686e-6 | 1.513e-5 | 1.032e-6 |
| Nonuniform | 45 | 5.661e-6 | 1.365e-5 | 6.827e-7 |
| Nonuniform | 90 | 5.248e-6 | 1.993e-5 | 1.784e-6 |

At 45 degrees, replacing CPML with PEC gives peak errors **27.12% uniform** and
**25.19% nonuniform**, confirming that the benchmark detects returned reflections.
At 1.5 ns, the CPML fields remain finite and interior Ez norms are 2.903e-5 and
2.536e-5 of their initial norms respectively. These are measured cases, not a
universal broadband/grazing-incidence reflection specification. R0 is a design
parameter, not the observed error. Additional bands, material interfaces near
collars, and extreme resolutions need dedicated dataset validation.

## Physical excitation and observation

Point `normalization="current"` takes integrated Jz in amperes. Bilinear weights
sum to one and are divided by dual-cell areas; the electric update multiplies by
`-dt/(epsilon*(1+sigma*dt/(2*epsilon)))`. Waveforms are sampled at half timesteps.
A test reconstructs the integrated injected current on two grids and two dt values
and verifies it matches the same analytic waveform. This does not remove numerical
dispersion or spatial sampling error.

`current_density` is per-node Jz in A/m². Legacy `field_increment` remains the default,
with nearest-node placement and waveform samples at electric-step endpoints; use
physical current for comparisons involving changing dt or point-cell area.

Point receivers use bilinear interpolation; fixed-count line receivers preserve
physical sample locations across meshes. The interpolation is exact for an analytic
bilinear test field. Four raw node histories per sample are recorded on GPU and
combined after download, so raw receiver history memory is four times the physical
sample history. `resample` linearly interpolates onto a common in-range time vector
and refuses extrapolation. Source/receiver stencil support is excluded from PML.

For dataset comparisons, fix the physical waveform, coordinates, and duration.
Use common temporal coverage, spectral frequencies, and windowing. The provided
post-run DFT is unwindowed, normalized by dt, and limited to Nyquist. Its raw run
endpoints can differ because Nt=ceil(T/dt); align/crop histories when exact common
windows matter. No dataset generator, convergence controller or training loop is
included in stage 2.

## Reproduce

```powershell
.\scripts\build_cuda.ps1
.venv\Scripts\python.exe -m pytest -q --basetemp=artifacts/pytest-stage2
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\python.exe examples\cnn_mesh.py
.venv\Scripts\python.exe examples\plot_anchor_grading.py
.venv\Scripts\python.exe benchmarks\validate_cpml.py
.venv\Scripts\python.exe examples\plot_cpml.py
```

Plot scripts emit PNG/SVG; the benchmark emits `artifacts/stage2/cpml_validation.json`.
Figures were visually inspected. Generated artifacts and demo weights are ignored
by Git; their reproduction code is committed. Checkpoints are now format v2 with
nine channels and the 1.4 grading/projection policy. Older checkpoints are rejected.
