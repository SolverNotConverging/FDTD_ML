# ScatterMesh

Nonuniform-grid plane-wave scattering for low-budget learned meshing.
This is a fresh project. The receiver-driven CNN experiments have been archived.

The current implementation is an independent **NumPy CPU reference solver** for
2D TMz dielectric scattering: nonuniform Yee updates, sampled material averaging,
conductivity, oblique broadband illumination, CPML, streaming surface DFT, and
near-to-far transformation. Experimental PEC rectangles/circles support cut edges
and cell enlargement without boundary anchors. Output is 2D scattering width in
metres. CUDA, mixed PEC/dielectric scenes, thin screens, reference campaigns, and
CNN training are subsequent milestones.

- [Implementation plan](IMPLEMENTATION_PLAN.md)
- [Measured progress and limitations](PROGRESS.md)
- [Numerical conventions](docs/numerics.md)
- [openEMS implementation review](docs/openems_review.md)
- [Conformal PEC and cell enlargement](docs/conformal_pec.md)
- [Simple-first training curriculum and joint loss](docs/curriculum.md)
- [Archived receiver-CNN project](archive/receiver_cnn_2026-09-22/README.md)

## Run the initial validation

The existing `.venv` is retained and has the required numerical/test packages.
Use `PYTHONPATH=src` for standalone scripts so imports select the new package.

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/qualify_scattering.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/pilot_conformal_pec.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/qualify_pec_cylinders.py
```

Outputs are `runs/scattering_bootstrap/report.json`, `spectra.npz`, and
`scattering_validation.png`. This bounded CPU check completes and exits; it does
not start a dataset/training worker. It compares 64/128 uniform and nonuniform
meshes at two incidence angles, followed by four sensitivity checks.
The separate PEC pilot writes `runs/conformal_pec_pilot/` with analytic cylinder
comparisons, off-grid geometry plots, and very small cut-fraction stability checks.
The restartable PEC-cylinder matrix writes `runs/pec_cylinder_qualification/` and
reuses a case only after checking its physics-source/configuration fingerprint and
saved complex arrays. See the [measured matrix report](docs/validation/pec_cylinder_qualification.md).

For a fresh environment, install this package and the tools:

```bash
python -m pip install -e . pytest matplotlib
```

## Example

```python
import numpy as np
from scattermesh import Circle, Grid, Material, PlaneWave, focused_axis, simulate

x = focused_axis(1.2, 128, width=0.12, strength=2)
result = simulate(
    Grid(x, x, max_ratio=3),
    [Circle((0.6, 0.6), 0.06, Material(epsilon_r=4))],
    PlaneWave(1e9, 1e-9, 9e-9, angle=np.pi/6, origin=(0.6, 0.6)),
    frequencies=[0.8e9, 1e9, 1.2e9], duration=30e-9, pml_thickness=0.15,
)
angles = np.linspace(0, 2*np.pi, 180, endpoint=False)
field = result.monitor.normalized_far_field(angles)  # complex128, sqrt(m); primary target
width = result.monitor.scattering_width(angles)  # derived [frequency, angle], metres
print(result.diagnostics)
```

`focused_axis` is an experiment helper, not a trained model or optimized teacher.
This prototype has no automatic reference-convergence label.

For PEC-only scenes, use `Circle(..., PEC())` or `Rectangle(..., PEC())` and select
`pec_mode="enlarged"` to test cell enlargement. The default `"conformal"` retains
small cut regions and reduces dt as required; `"staircase"` is a comparison mode.
The enlarged method remains experimental and computes its own stability bound.

Both pilot scripts preserve complex far fields and incident spectra in `spectra.npz`.
Phase origins and the Fourier convention are recorded in the reports. Future
training includes complex real/imaginary error and a floored log-RCS loss. The
initial data curriculum starts with one PEC cylinder, then one dielectric cylinder.

## Archive and restoration

The complete previous source state is commit `5878ddd` on
`codex/archive-receiver-cnn-20260922`. The active branch is
`codex/plane-wave-scattering`.

Old files live in `archive/receiver_cnn_2026-09-22/`. The large local artifacts and
compiled outputs are preserved there but excluded from Git, as they were before
archiving. `INVENTORY.json` records SHA256 checksums; `INVENTORY_SUMMARY.json`
records counts and link checks. This is a local archive, not an off-machine backup.

The root `artifacts` compatibility symlink points to the archived artifact tree.
Keep it when following historical absolute checkpoint links. On another checkout,
restore the local artifact archive separately and create that link if needed.
For old source, create a separate worktree at the archive branch; the source
snapshot alone does not contain the ignored datasets or Python environments.
