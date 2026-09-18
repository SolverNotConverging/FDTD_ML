from fdtdmesh import FDTD_2D_Ez

sim = FDTD_2D_Ez(0.02, 0.015, Nx=80, Ny=60, f_max=20e9, t_end=1e-9)
glass = sim.add_material("glass", epsilon_r=4)
sim.add_rectangle(glass, x_position=(0.007, 0.012), y_position=(0.003, 0.012))
sim.add_source("point", x=0.003, y=0.0075, width=2e-11, delay=8e-11)
rx = sim.add_receiver("point", x=0.016, y=0.0075)
result = sim.run()
print(result.diagnostics)
print("Peak receiver Ez:", abs(result.receivers[rx]).max())
