import numpy as np

from fdtdmesh import AxisConstraints, FDTD_2D_Ez

sim = FDTD_2D_Ez(0.02, 0.015, 80, 60, 20e9, t_end=1e-9)
sim.add_pec_line(x=0.011, y=(0.005, 0.010))
sim.add_source("point", x=0.003, y=0.007, width=2e-11, delay=8e-11)
sim.add_receiver("point", x=0.017, y=0.007)
u = (np.arange(128) + 0.5) / 128
sim.mesh_from_density(
    1 + 4 * np.exp(-(((u - 0.55) / 0.12) ** 2)),
    np.ones(128),
    x_constraints=AxisConstraints(min_spacing=5e-5, max_ratio=1.5),
)
print(sim.run().diagnostics)
