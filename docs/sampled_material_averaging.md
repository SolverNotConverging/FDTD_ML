# Sampled electric-material averaging

The TMz solver now supports `material_averaging="sampled"` for nonmagnetic
isotropic scenes (`mu_r=1`, `sigma_h=0`). The legacy API/evaluator default remains
`"point"`; newly launched quota campaigns default to `"sampled"`. The existing
ten-scene pilot retains its saved point-sampling configuration.

At each Ez site, the integration rectangle extends to the midpoints of adjacent
mesh lines, clipped at the physical domain. Its physical area is respected on
nonuniform meshes. Deterministic midpoint subcell samples apply the same
last-primitive-wins geometry priority as the scene. Vacuum participates normally.
The implementation averages epsilon_r and sigma_e, then builds the existing
Crank–Nicolson electric-loss coefficients and current-source scaling. Magnetic
coefficients and CUDA stepping kernels are unchanged.

Polygon-segment/dual-cell intersection and circle distance bounds select cells
that might contain interfaces; matching corner labels alone cannot hide an
entire enclosed object. Defaults use 8×8, then 16×16, and where necessary 32×32
samples. Successive effective-parameter estimates are compared at relative
1e-3 tolerance, with floors of 1 for epsilon_r and 1e-12 S/m for sigma_e.
Sample batches are bounded in size, and counts are capped at 256 per axis.

Successive quadrature agreement is a heuristic, not a rigorous error bound:
fixed subcell samples can alias a boundary. Fixed-N sampling does not guarantee
second-order convergence. Diagnostics report candidate cells, cells reaching
the sampling limit without agreement, and whether comparison was performed.
Reference mesh convergence is still required. A separate higher-sample run
checks whether quadrature error matters for the observable.

PEC is not a finite permittivity or conductivity to blend. Its binary mask and
anchored thin lines are unchanged. We conservatively retain point treatment
where a dual cell touches a PEC boundary or lies inside any original PEC body,
including bodies later partially overwritten by ordinary geometry. A defensive
sample-level PEC check also falls back to point treatment. This intentionally
leaves PEC staircasing unresolved; there is no conformal PEC implementation here.

## Use

```python
simulation = FDTD_2D_Ez(
    Lx, Ly, Nx, Ny, f_max,
    material_averaging="sampled",
    averaging_samples=8,
    averaging_max_samples=32,
    averaging_tolerance=1e-3,
)
```

The same four options exist on `EvaluationConfig`, and as hyphenated arguments
on `python -m fdtdmesh.data references`, `evaluate`, and
`python -m fdtdmesh.data.campaign`. Use `--material-averaging point` for a baseline.
Set both sample counts to the same value for fixed quadrature. Averaged reference
identities include the method and quadrature settings. Legacy point-reference
identities omit these inactive options to preserve compatibility. Do not reuse
point references as averaged references or mix their convergence sequences.

## Measured qualification (2026-09-19)

The controlled cavity is 20×15 mm with a planar interface at x/Lx=0.371, vacuum
on one side and the stated dielectric on the other. An independent continuum
transfer-matrix root supplies the resonance frequency (and decay for the lossy
case). The discrete mode calculation uses production coefficients and includes
leapfrog/CN temporal dispersion. These are frequency errors, not waveform errors.

| 512×512 cavity | Point error | 32×32 sampled error | Improvement |
|---|---:|---:|---:|
| epsilon_r=12, sigma_e=0 | 0.046810% | 0.001552% | 30.2× |
| epsilon_r=30, sigma_e=0 | 0.048121% | 0.001597% | 30.1× |
| epsilon_r=12, sigma_e=0.1 S/m | 0.046828% | 0.001553% | 30.2× |

The lossy decay-rate error fell from 0.004546% to 0.000128%. This sampled result
is less accurate than exact analytic filling fractions in the earlier experiment;
it must not be described as reproducing that experiment's exact averaging.

For actual pilot scene `train-v5-000000`, same scene, pulse, physical duration,
float64 fields, PML and meshes, the 512→1024 comparison was:

| Maximum receiver metric | Point | Adaptive 8–32 sampled |
|---|---:|---:|
| Waveform L2 difference | 3.4256% | 3.3948% |
| Spectrum L2 difference | 3.5551% | 3.5294% |

Both methods still fail the 2% spatial threshold. No 2048 or 4096 averaged run
was used in this comparison, and no converged averaged reference is claimed.
At mesh 512, replacing adaptive 8–32 sampling by fixed 64×64 changes waveforms by
0.00692% and spectra by 0.01280%, far below the observed spatial differences.
The controlled cases establish a dielectric-mapping improvement. The actual
scene establishes only marginal benefit, so PEC boundaries/corners and other
errors need separate diagnosis before claiming a campaign convergence fix.

Reproduce with `scripts/qualify_sampled_averaging.py` and
`scripts/compare_sampled_pilot.py`. Local run configurations, JSON metrics,
waveforms, plots and test logs are under `artifacts/sampled_averaging/`.
The GPU diagnostics shared GPU 3 with the pilot; the contention interval is
recorded in `artifacts/reference_v5_pilot10/diagnostic_contention.json`. Pilot
wall times overlapping this interval are not uncontended performance results.

A subsequent [PEC-face anchor experiment](pec_anchor_experiment.md) improves the
same pilot scene further, but still does not meet the 2% spatial threshold at
1024. This is a separate mesh-policy experiment with material averaging enabled.
