import numpy as np
import pytest

from fdtdmesh import FDTD_2D_Ez, Scene2D
from fdtdmesh.solver.coefficients import build_coefficients


def test_shapes_priority_and_materials():
    s = Scene2D(1, 1)
    dielectric = s.add_material("glass", epsilon_r=4, mu_r=2, sigma_e=0.1)
    s.add_rectangle(dielectric, (0.1, 0.9), (0.1, 0.9))
    s.add_circle("PEC", (0.5, 0.5), 0.2)
    s.add_triangle("vacuum", [(0.45, 0.45), (0.55, 0.45), (0.5, 0.6)])
    eps, mu, sigma, pec = s.sample([0.2, 0.5, 0.95], [0.2, 0.5, 0.95])
    assert eps[0, 0] == 4 and mu[0, 0] == 2 and sigma[0, 0] == 0.1
    assert eps[1, 1] == 1 and not pec[1, 1]
    assert eps[-1, -1] == 1
    assert s.sample([0.4], [0.5])[3][0, 0]


def test_thin_pec_and_stale_mesh():
    s = FDTD_2D_Ez(1, 1, 20, 20, 1e9, Nt=2)
    s.mesh_uniform()
    s.add_pec_line(x=0.413, y=(0.2, 0.8))
    with pytest.raises(ValueError, match="missing mandatory"):
        s._prepare()
    s.mesh_from_density([1], [1])
    c, *_ = s._prepare()
    i = np.flatnonzero(s.mesh.x == 0.413)[0]
    assert c.pec[i, (s.mesh.y >= 0.2) & (s.mesh.y <= 0.8)].all()


def test_fixed_duration_cfl_and_unsafe_dt():
    a = FDTD_2D_Ez(1, 1, 20, 20, 1e9, t_end=1e-8)
    b = FDTD_2D_Ez(1, 1, 20, 20, 1e9, t_end=1e-8)
    a.mesh_uniform()
    b.mesh_from_density([1, 10], [1])
    ca, _, _, wa, *_ = a._prepare()
    cb, _, _, wb, *_ = b._prepare()
    assert cb.dt < ca.dt and len(wb) > len(wa)
    assert len(wa) * ca.dt >= a.t_end and (len(wa) - 1) * ca.dt < a.t_end
    with pytest.raises(ValueError, match="CFL"):
        build_coefficients(a, a.mesh, dt=ca.dt_cfl * 1.01)


def test_source_on_pec_rejected_and_overlap_merged():
    s = FDTD_2D_Ez(1, 1, 10, 10, 1e9, Nt=2)
    s.add_source("point", x=0.5, y=0.5, amplitude=1)
    s.add_source("point", x=0.5, y=0.5, amplitude=2)
    _, _, indices, waves, *_ = s._prepare()
    assert len(indices) == 1 and waves.shape == (2, 1)
    s.add_source("point", x=0, y=0.5)
    with pytest.raises(ValueError, match="PEC"):
        s._prepare()


def test_uniform_refuses_unaligned_anchor():
    s = FDTD_2D_Ez(1, 1, 10, 10, 1e9, Nt=2)
    s.add_anchor("x", 0.333)
    with pytest.raises(ValueError, match="Uniform mesh"):
        s.mesh_uniform()


def test_loss_coefficients_and_magnetic_material():
    s = FDTD_2D_Ez(1, 1, 10, 10, 1e9, Nt=2)
    mat = s.add_material("loss", epsilon_r=4, mu_r=2, sigma_e=0.01)
    s.add_rectangle(mat, (0, 1), (0, 1))
    s.mesh_uniform()
    c, *_ = s._prepare()
    assert (c.ca < 1).all() and (c.ca > -1).all()


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.config("cpu"),
        lambda s: s.add_PML(8),
        lambda s: s.set_mesh([0, 0.5, 1], [0, 0.5, 1]),
    ],
)
def test_unsupported_or_wrong_budget(call):
    s = FDTD_2D_Ez(1, 1, 10, 10, 1e9, Nt=2)
    with pytest.raises((ValueError, NotImplementedError)):
        call(s)
