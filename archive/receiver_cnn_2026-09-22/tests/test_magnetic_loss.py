import numpy as np
import pytest

from fdtdmesh import FDTD_2D_Ez, Material
from fdtdmesh.constants import C0, EPS0, MU0
from fdtdmesh.ml import CHANNELS, ResUNet, load_model, rasterize, save_model
from fdtdmesh.solver.reference_tmz import run_reference
from fdtdmesh.solver.tmz import cuda_available, cuda_backend


def test_magnetic_material_priority_and_model_contract(tmp_path):
    with pytest.raises(ValueError):
        Material("bad", sigma_h=-1)
    s = FDTD_2D_Ez(1, 1, 16, 16, 1e9)
    s.add_material("magnetic", mu_r=30, sigma_h=12)
    s.add_rectangle("magnetic", (0.1, 0.9), (0.1, 0.9))
    s.add_rectangle("vacuum", (0.4, 0.6), (0.4, 0.6))
    _, mu, _, sh, _ = s.sample([0.2, 0.5], [0.2, 0.5], magnetic_loss=True)
    assert sh[0, 0] == 12 and sh[1, 1] == 0 and mu[0, 0] == 30
    assert len(s.sample([0.2], [0.2])) == 4  # Existing public sampling contract.
    image = rasterize(s, (16, 16), 1e9)
    assert image.shape == (10, 16, 16) and image[9].max() > 0
    path = tmp_path / "model.pt"
    save_model(path, ResUNet(16))
    model, metadata = load_model(path)
    assert model.width == 16 and metadata["format_version"] == 3
    assert metadata["input_channels"] == CHANNELS


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dual_loss_cpml_cuda_numpy(dtype):
    s = FDTD_2D_Ez(0.02, 0.02, 32, 32, 20e9, Nt=160, dtype=dtype)
    s.add_PML(pml_width=4, thickness=0.0025)
    s.add_material("loss", epsilon_r=12, mu_r=21, sigma_e=0.2, sigma_h=15000)
    s.add_rectangle("loss", (0.008, 0.014), (0.006, 0.015))
    s.add_source("point", x=0.005, y=0.01, width=1e-11, delay=4e-11)
    s.add_receiver("point", x=0.016, y=0.011)
    s.mesh_uniform()
    c, fields, sites, waves, receivers, *_ = s._prepare()
    expected = run_reference(c, fields, sites, waves, receivers)
    got = cuda_backend().run(c, fields, sites, waves, receivers)
    for a, b in zip(got[:4], expected):
        np.testing.assert_allclose(
            a,
            b,
            rtol=3e-5 if dtype == "float32" else 1e-11,
            atol=1e-6 if dtype == "float32" else 1e-13,
        )
    assert got[4]["stepping_transfers"] == 0


def _matched_cavity(n, backend):
    lx, ly = 0.02, 0.015
    eps, mu, gamma = 4.0, 7.0, 2e9
    omega = C0 / np.sqrt(eps * mu) * np.pi * np.sqrt(lx**-2 + ly**-2)
    s = FDTD_2D_Ez(lx, ly, n, n, 20e9, t_end=1.2 * 2 * np.pi / omega, dtype="float64")
    s.add_material(
        "fill", epsilon_r=eps, mu_r=mu, sigma_e=gamma * EPS0 * eps, sigma_h=gamma * MU0 * mu
    )
    s.add_rectangle("fill", (0, lx), (0, ly))
    s.mesh_uniform()
    c, fields, sites, waves, receivers, *_ = s._prepare()
    x, y = s.mesh.x, s.mesh.y
    xc, yc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    e = np.sin(np.pi * x[:, None] / lx) * np.sin(np.pi * y[None, :] / ly)
    t = -c.dt / 2
    hx = (
        -np.pi
        / (MU0 * mu * omega * ly)
        * np.sin(np.pi * x[:, None] / lx)
        * np.cos(np.pi * yc[None, :] / ly)
        * np.sin(omega * t)
        * np.exp(-gamma * t)
    )
    hy = (
        np.pi
        / (MU0 * mu * omega * lx)
        * np.cos(np.pi * xc[:, None] / lx)
        * np.sin(np.pi * y[None, :] / ly)
        * np.sin(omega * t)
        * np.exp(-gamma * t)
    )
    output = backend(c, [e, hx, hy], sites, waves, receivers)
    final = len(waves) * c.dt
    exact = e * np.exp(-gamma * final) * np.cos(omega * final)
    return np.linalg.norm(output[0] - exact) / np.linalg.norm(e)


def test_analytic_matched_electric_magnetic_cavity_numpy():
    coarse, fine = _matched_cavity(20, run_reference), _matched_cavity(40, run_reference)
    assert fine < coarse * 0.4 and fine < 0.01, (coarse, fine)


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_analytic_matched_electric_magnetic_cavity_cuda():
    coarse, fine = _matched_cavity(20, cuda_backend().run), _matched_cavity(40, cuda_backend().run)
    assert fine < coarse * 0.4 and fine < 0.01, (coarse, fine)
