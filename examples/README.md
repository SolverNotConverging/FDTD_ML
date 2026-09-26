# Mesh CNN simulation examples

Each script builds a continuous 2D TMz scene, loads the current trained mesh CNN,
projects its x/y density profiles to an exact square cell budget, and runs the
**compiled CUDA FDTD solver**. It saves a mesh PNG, a polar scattering plot, raw
simulation records, and `summary.json` under `runs_v2/examples/`. The polar
quantity is the **2D scattering width** (`2π|far field|²`), the RCS analogue for
this 2D model. The scripts save figures and never call `plt.show()`.

Run from the project root with the active local environment:

```bash
.venv/bin/python examples/circle.py --budget 64 --device cuda:0
.venv/bin/python examples/aircraft.py --budget 96 --compare --device cuda:1
.venv/bin/python examples/holes.py --budget 64 --compare --max-reference-cells 1024
```

`--compare` first runs progressively finer **uniform** reference grids until
successive complex fields change by at most 1% and scattering widths by at most
2%, with the field tail settled. It then runs CNN and uniform meshes at the same
requested budget and saves `mesh_comparison.png` and
`rcs_comparison_polar.png`. If the reference cannot meet these conditions before
`--max-reference-cells`, the script saves `reference_report.json` and stops; it
does not label an unresolved result as converged. These spatial checks are a
practical example reference, not a full independent numerical qualification.

The scripts use the frozen C0 checkpoint when one is available, otherwise the
best saved C0 validation-profile checkpoint. Until the current C0 training
produces one, pass `--checkpoint /absolute/path/to/checkpoint.pt` to use a
compatible mesh-CNN v2 checkpoint. Archived receiver/ResU checkpoints are not
compatible with this model.

| Script | Geometry |
| --- | --- |
| `circle.py` | dielectric circle |
| `ellipse.py` | rotated dielectric ellipse |
| `rectangle.py` | dielectric rectangle |
| `gaps.py` | two dielectric circles with a 25 mm gap |
| `concave.py` | open U-shaped cavity |
| `holes.py` | plate with two holes |
| `ring.py` | square ring |
| `bird.py` | approximate bird outline with lossy dielectric |
| `human.py` | approximate human outline with lossy dielectric |
| `aircraft.py` | approximate aircraft outline with dielectric material |
| `warship.py` | approximate warship outline with dielectric material |

The current training stage is C0. Later shapes are demonstrations of the
inference and solver path; their CNN mesh accuracy needs stage-specific
validation before being treated as a scientific result. The engineered outlines
are simple 2D silhouettes rather than detailed metal CAD models. A simulation
that cannot settle or has an incompatible geometry/grid combination keeps that
status in its saved record.

Common options: `--budget` (32, 48, 64, 96, 128, 192, 256, 384, or 512 cells
per axis), `--angle-deg`, `--frequency-ghz` (0.8, 1.0, or 1.2 for the polar plot),
`--output`, `--checkpoint`, `--reference-complex-tol`, and
`--reference-width-tol`. All three frequencies are simulated in each run.
