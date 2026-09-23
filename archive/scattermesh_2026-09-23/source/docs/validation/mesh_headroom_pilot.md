# Low-budget mesh-headroom pilot

The first M3 search asks whether smooth interface-focused tensor grids improve PEC-
cylinder scattering at equal computational cost. Four analytic-reference scenes span
two radii, subcell positions, and axial/oblique/reverse incidence. Each uses 32, 48,
64, or 96 cells per axis and six mesh candidates: true uniform plus five unanchored
Gaussian interface-density projections with grading limits 1.4, 2, or 3.

Every candidate uses the same 50 ns physical duration. Accuracy is the provisional
joint complex-far-field and floored log-scattering-width loss. Cost is the measured
`Nx*Ny*Nt`, so a small CFL step is charged directly. The analytic PEC-cylinder field
retains absolute translation and incidence phase.

## Result

The run completed 96 candidates on four GPUs. Seventy-seven passed the `1e-5`
tail gate, fourteen were rejected because the coarse tensor grid allowed a circle to
split one Yee edge twice, and five aggressive 32-cell grids remained unsettled at
the declared 50 ns pilot limit. All sixteen uniform baselines settled.

No accepted interface-focused mesh displaced a uniform mesh on any scene's
error-versus-update Pareto frontier. Relative to the uniform mesh with the same
cell count, accepted focused candidates used 1.39--3.61 times as many updates and
had 1.25--5.44 times the joint loss. The closest case was the wide-focus 96-cell
large cylinder: 1.25 times the loss at 1.39 times the update cost.

This is a negative result for this candidate family, not evidence that nonuniform
meshing has no value. Concentrating tensor-grid lines only at a circular interface
coarsens the propagation, observation, and absorbing regions while reducing the
global CFL step. The next search must include object-region, hybrid, and randomized
density candidates before the low-budget-headroom gate can pass. CNN training must
not use the current interface candidates as teacher targets.

## Expanded search and dense controls

The follow-up search added object-region, hybrid region/interface, and two
deterministic randomized densities at every scene and primary budget, for 208 total
candidates. Of these, 173 settled, 16 were geometrically infeasible, and 19 missed
the 50 ns tail gate. Mild region focus initially added apparent Pareto points for the
medium and large cylinders when uniform controls were spaced only at 32/48/64/96.

We therefore added 96 uniform controls: every four cells through 128, plus 65--67
around the only remaining close comparison. The large-cylinder 64-cell wide-region
candidate used 6.12 million updates with joint loss `3.0125e-4`. A 67-cell uniform
grid used fewer updates, 5.91 million, and achieved lower loss, `2.5629e-4`.
After this refinement every Pareto point is uniform in all four scenes.

The M3 headroom gate is therefore **not passed for isolated PEC cylinders**. This
is useful curriculum evidence: the analytic PEC family remains a solver and loss
validation family, but it should not dominate mesh-policy supervision. The next
simple headroom search moves to dielectric cylinders, where internal wavelength and
material averaging create a physical reason to redistribute resolution.

The initial 35 ns attempt is retained locally: 18 candidates failed settling, which
triggered the common-duration rerun rather than selective time extensions. Reproduce
the restartable four-way search and summary with:

```bash
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/pilot_mesh_headroom.py \
  --device cuda:0 --shard 0 --shards 4
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/pilot_mesh_headroom.py --summarize
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/run_uniform_headroom_controls.py \
  --device cuda:0 --shard 0 --shards 4
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/run_uniform_headroom_controls.py --summarize
```

Local records, complex spectra, report, retained 35 ns attempt summary, and plot are
under `runs/mesh_headroom_pilot/`. Expanded-search and dense-control artifacts are
under `runs/mesh_headroom_expanded/`.
