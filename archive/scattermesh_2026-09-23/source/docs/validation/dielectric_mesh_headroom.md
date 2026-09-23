# Simple dielectric mesh-headroom qualification

This M3 search tests whether nonuniform tensor grids can improve analytic-cylinder
scattering accuracy at the same measured FDTD update budget. Four simple scenes
cover epsilon_r 2 and 4, two scales, multiple subcell locations and incidence
angles, and one sigma_e=0.05 S/m case. Mu_r remains 1 and sigma_h remains 0.

The candidate sweep uses 32, 48, 64, and 96 cells per axis with true uniform,
interface, object-region, hybrid, and deterministic randomized densities. Every
run uses the same 70 ns duration, three frequencies, 180 observation angles,
sampled material filling fractions, and the joint complex-field plus floored
log-scattering-width loss. Cost is the measured `Nx*Ny*Nt`.

The initial sweep contains 208 candidates. Dense comparison adds 156 uniform
controls, including every integer resolution around each close low-budget crossing
and four-cell spacing through 128. Of 364 total runs, 355 pass the `1e-5` tail gate.
The nine rejected runs are all epsilon_r=4 uniform/random 32--39-cell cases that do
not settle within 70 ns; they are excluded rather than ranked.

## Matched-update result

Nonuniform headroom is small for epsilon_r=2 and substantial for epsilon_r=4.

| Scene/candidate | Candidate loss / updates | Best affordable uniform | Uniform loss / updates | Advantage |
|---|---:|---|---:|---:|
| eps_r=2 medium, 64² wide region | 1.2977e-3 / 9.18M | 69² | 1.3275e-3 / 9.03M | 1.023x |
| eps_r=4 medium, 64² medium region | 2.6123e-3 / 11.65M | 75² | 4.9534e-3 / 11.59M | **1.896x** |
| eps_r=4 medium, 96² wide hybrid | 6.6606e-4 / 36.15M | 108² | 1.2406e-3 / 34.62M | **1.863x** |
| lossy eps_r=4, 64² wide region | 2.4818e-3 / 9.36M | 68² | 3.5925e-3 / 8.64M | **1.448x** |

The strongest 64² epsilon_r=4 comparison improves both saved loss components:
complex MSE falls from `4.0779e-3` to `2.3120e-3`, while scaled log-width loss
falls from `3.5018e-3` to `1.2012e-3`. The candidate has actual grading only 1.058;
the gain does not rely on abrupt ratio-3 transitions. Its tail ratio is `3.4e-9`.

These results pass the low-budget-headroom gate for moderate-contrast dielectric
cylinders. They do not establish a universal learned mesher: the search uses four
scenes and hand-parameterized densities. The next stage can generate a larger
simple-cylinder pool with grouped train/validation/test lineages and retain multiple
Pareto candidates per geometry, illumination, and budget. PEC-only cylinders remain
analytic validation cases rather than dominant teacher samples.

Reproduce the restartable sweep and dense controls with:

```bash
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/pilot_dielectric_mesh_headroom.py \
  --device cuda:0 --shard 0 --shards 4
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/run_dielectric_uniform_headroom_controls.py \
  --device cuda:0 --shard 0 --shards 4
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/run_dielectric_uniform_headroom_controls.py --summarize
```

Local complex spectra, records, reports, and plots are under
`runs/dielectric_mesh_headroom/`.
