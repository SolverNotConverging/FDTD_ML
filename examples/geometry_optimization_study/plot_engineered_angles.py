"""Show exact engineered CSG silhouettes at three incidence orientations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from study_cases import make_simulation

from fdtdmesh.benchmarks.engineered_shapes import ENGINEERED_SHAPES

out = Path("docs/figures/geometry_optimization/engineered_rotations.png")
out.parent.mkdir(parents=True, exist_ok=True)
angles = (0, 45, 90)
fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
for row, shape in enumerate(ENGINEERED_SHAPES):
    for col, angle in enumerate(angles):
        sim = make_simulation(shape, angle, 1.0)
        ax = axes[row, col]
        sim.plot_geometry(units="wavelength", ax=ax)
        a, b, c, d = sim.layout.tfsf_box
        lam = sim.wavelength
        pad = 0.1 * max(b - a, d - c)
        ax.set(
            xlim=((a - pad) / lam, (b + pad) / lam),
            ylim=((c - pad) / lam, (d + pad) / lam),
            title=f"{shape.replace('_', ' ').title()} · {angle}°",
        )
        legend = ax.get_legend()
        if legend:
            legend.remove()
fig.savefig(out, dpi=170)
plt.close(fig)
print(out)
