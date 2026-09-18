import numpy as np
import pytest

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.constants import C0, EPS0, MU0
from fdtdmesh.ml import ResUNet, save_model
from fdtdmesh.solver.reference_tmz import run_reference
from fdtdmesh.solver.tmz import cuda_available, cuda_backend

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_cuda_matches_numpy(dtype, nonuniform):
    sim = FDTD_2D_Ez(0.02, 0.015, 32, 24, 20e9, Nt=180, dtype=dtype)
    mat = sim.add_material("lossy", epsilon_r=3.5, mu_r=1.3, sigma_e=0.04)
    sim.add_rectangle(mat, (0.007, 0.013), (0.002, 0.013))
    sim.add_circle("PEC", (0.016, 0.007), 0.001)
    sim.add_source("point", x=0.003, y=0.007, width=1e-11, delay=4e-11)
    sim.add_source("point", x=0.003, y=0.007, width=1e-11, delay=4e-11, amplitude=0.5)
    sim.add_receiver("point", x=0.012, y=0.007)
    sim.add_line_monitor(x=0.005, y=(0.00375, 0.01125))
    if nonuniform:
        sim.mesh_from_density([1, 2, 4, 2, 1], [1, 3, 2, 1])
    else:
        sim.mesh_uniform()
    c, fields, sources, waves, receivers, *_ = sim._prepare()
    ref = run_reference(c, fields, sources, waves, receivers)
    gpu = cuda_backend().run(c, fields, sources, waves, receivers)
    for actual, expected in zip(gpu[:4], ref):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2e-5 if dtype == "float32" else 1e-11,
            atol=2e-6 if dtype == "float32" else 1e-13,
        )
    assert np.max(abs(gpu[3])) > 1e-5
    assert np.all(gpu[0][c.pec.astype(bool)] == 0)
    assert gpu[4]["stepping_transfers"] == 0


def cavity_error(n, nonuniform=False, epsilon_r=1.0, mu_r=1.0, sigma_e=0.0):
    lx, ly = 0.02, 0.015
    omega0 = C0 / np.sqrt(epsilon_r * mu_r) * np.pi * np.sqrt(lx**-2 + ly**-2)
    decay = sigma_e / (2 * EPS0 * epsilon_r)
    omega = np.sqrt(omega0**2 - decay**2)
    sim = FDTD_2D_Ez(lx, ly, n, n, 20e9, t_end=1.3 * 2 * np.pi / omega, dtype="float64")
    material = sim.add_material("fill", epsilon_r=epsilon_r, mu_r=mu_r, sigma_e=sigma_e)
    sim.add_rectangle(material, (0, lx), (0, ly))
    if nonuniform:
        u = np.linspace(0, 1, n + 1)
        sim.set_mesh(
            lx * (u + 0.06 * np.sin(2 * np.pi * u)), ly * (u + 0.04 * np.sin(2 * np.pi * u))
        )
    else:
        sim.mesh_uniform()
    c, *_ = sim._prepare()
    x, y = sim.mesh.x, sim.mesh.y
    xc, yc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    e0 = np.sin(np.pi * x[:, None] / lx) * np.sin(np.pi * y[None, :] / ly)
    # Leapfrog initial H is at -dt/2 for the exact standing-wave solution.
    hx = (
        -np.pi
        / (MU0 * mu_r * omega * ly)
        * np.sin(np.pi * x[:, None] / lx)
        * np.cos(np.pi * yc[None, :] / ly)
        * np.sin(-omega * c.dt / 2)
        * np.exp(decay * c.dt / 2)
    )
    hy = (
        np.pi
        / (MU0 * mu_r * omega * lx)
        * np.cos(np.pi * xc[:, None] / lx)
        * np.sin(np.pi * y[None, :] / ly)
        * np.sin(-omega * c.dt / 2)
        * np.exp(decay * c.dt / 2)
    )
    out = sim.run(initial_fields={"Ez": e0, "Hx": hx, "Hy": hy})
    time = out.times[-1]
    exact = (
        e0 * np.exp(-decay * time) * (np.cos(omega * time) - decay / omega * np.sin(omega * time))
    )
    return np.linalg.norm(out.fields["Ez"] - exact) / np.linalg.norm(e0)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_analytic_cavity_convergence(nonuniform):
    coarse = cavity_error(20, nonuniform)
    fine = cavity_error(40, nonuniform)
    assert fine < coarse * 0.4 and fine < 0.01, (coarse, fine)


@pytest.mark.parametrize("sigma", [0, 0.04])
def test_material_cavity_analytic_convergence(sigma):
    coarse = cavity_error(20, True, epsilon_r=4, mu_r=2, sigma_e=sigma)
    fine = cavity_error(40, True, epsilon_r=4, mu_r=2, sigma_e=sigma)
    assert fine < coarse * 0.4 and fine < 0.01, (coarse, fine)


def test_pec_wall_blocks_transmission():
    sim = FDTD_2D_Ez(0.02, 0.02, 40, 40, 20e9, Nt=250)
    sim.add_pec_line(x=0.01, y=(0, 0.02))
    sim.add_source("point", x=0.005, y=0.01, width=1e-11, delay=4e-11)
    left = sim.add_receiver("point", x=0.006, y=0.01)
    right = sim.add_receiver("point", x=0.015, y=0.01)
    out = sim.run()
    assert np.max(abs(out.receivers[left])) > 0.01
    assert np.max(abs(out.receivers[right])) == 0
    assert np.all(out.fields["Ez"][20] == 0)


def test_cnn_to_cuda_end_to_end(tmp_path):
    checkpoint = tmp_path / "model.pt"
    save_model(checkpoint, ResUNet(4), raster_shape=(24, 32))
    sim = FDTD_2D_Ez(0.02, 0.015, 30, 24, 20e9, t_end=2e-10)
    sim.add_source("line-soft", x=0.004, y=(0.004, 0.011), width=1e-11, delay=4e-11)
    rx = sim.add_receiver("point", x=0.014, y=0.007)
    sim.mesh_with_model(checkpoint)
    a, b = sim.run(), sim.run()
    assert a.receivers[rx].shape[0] == a.diagnostics["Nt"]
    assert np.max(abs(a.receivers[rx])) > 1e-5
    assert a.diagnostics["cell_updates"] == 30 * 24 * a.diagnostics["Nt"]
    assert a.diagnostics["stepping_transfers"] == 0
    np.testing.assert_array_equal(a.fields["Ez"], b.fields["Ez"])
    assert a.spectrum(rx, [1e9, 2e9]).shape == (2, 1)


def test_residency_empty_monitors_and_zero_state():
    sim = FDTD_2D_Ez(0.01, 0.01, 12, 10, 20e9, Nt=3)
    a = sim.run()
    sim.Nt = 100
    b = sim.run()
    assert a.diagnostics["h2d_bytes"] == b.diagnostics["h2d_bytes"]
    assert a.diagnostics["d2h_bytes"] == b.diagnostics["d2h_bytes"]
    assert a.diagnostics["stepping_transfers"] == b.diagnostics["stepping_transfers"] == 0
    assert np.max(abs(b.fields["Ez"])) == 0 and b.receivers == {}


def test_runtime_rejects_malformed_array_before_native_call():
    sim = FDTD_2D_Ez(0.01, 0.01, 12, 10, 20e9, Nt=3)
    c, f, s, w, r, *_ = sim._prepare()
    f[1] = f[1][:-1]
    with pytest.raises(ValueError, match="shapes"):
        cuda_backend().run(c, f, s, w, r)


@pytest.mark.parametrize("indices", [[2**32], [1.5], [-1]])
def test_runtime_indices_do_not_wrap_or_truncate(indices):
    sim = FDTD_2D_Ez(0.01, 0.01, 12, 10, 20e9, Nt=3)
    c, f, s, w, r, *_ = sim._prepare()
    with pytest.raises(ValueError, match="indices"):
        cuda_backend().run(c, f, s, w, indices)
