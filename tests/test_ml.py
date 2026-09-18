import numpy as np
import pytest
import torch

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.ml import ResUNet, conditioning, load_model, pool_axes, rasterize, save_model


def test_network_shape_positivity_and_gradients():
    torch.manual_seed(2)
    model = ResUNet(width=4)
    logits = model(torch.randn(2, 8, 17, 23), torch.randn(2, 5))
    x, y = pool_axes(logits)
    assert x.shape == (2, 23) and y.shape == (2, 17) and (x > 0).all() and (y > 0).all()
    (x.square().mean() + y.square().mean()).backward()
    assert model.enc1.conv1.weight.grad.abs().sum() > 0
    assert model.bottom.film.weight.grad.abs().sum() > 0


def test_pool_axis_orientation_and_constant_raster_invariance():
    logits = torch.zeros(1, 2, 9, 13)
    logits[:, 0, :, 7] = 3
    logits[:, 1, 2, :] = 4
    x, y = pool_axes(logits)
    assert x.argmax() == 7 and y.argmax() == 2
    a, _ = pool_axes(torch.ones(1, 2, 8, 11))
    b, _ = pool_axes(torch.ones(1, 2, 30, 11))
    torch.testing.assert_close(a, b)


def test_checkpoint_roundtrip_and_mesh(tmp_path):
    torch.manual_seed(4)
    model = ResUNet(4).eval()
    path = tmp_path / "mesher.pt"
    save_model(path, model, raster_shape=(17, 23))
    loaded, metadata = load_model(path)
    inputs, cond = torch.randn(1, 8, 17, 23), torch.randn(1, 5)
    torch.testing.assert_close(model(inputs, cond), loaded(inputs, cond))
    sim = FDTD_2D_Ez(0.02, 0.015, 30, 20, 20e9, Nt=5)
    sim.add_pec_line(x=0.011, y=(0.004, 0.009))
    mesh = sim.mesh_with_model(path)
    assert mesh.Nx == 30 and mesh.Ny == 20 and 0.011 in mesh.x
    assert 0.004 in mesh.y and 0.009 in mesh.y
    assert metadata["training_commit"] == "untrained"
    data = torch.load(path, weights_only=True)
    data["format_version"] = 999
    torch.save(data, path)
    with pytest.raises(ValueError, match="format_version"):
        load_model(path)


def test_scaling_and_raw_raster():
    a = FDTD_2D_Ez(0.02, 0.01, 20, 10, 20e9, Nt=2)
    b = FDTD_2D_Ez(0.2, 0.1, 20, 10, 2e9, Nt=2)
    for sim in (a, b):
        sim.add_pec_line(x=sim.Lx * 0.413, y=(sim.Ly * 0.2, sim.Ly * 0.8))
        sim.add_source("point", x=sim.Lx * 0.5, y=sim.Ly * 0.2)
        sim.add_receiver("line", x=sim.Lx * 0.5, y=(sim.Ly * 0.2, sim.Ly * 0.8))
    np.testing.assert_allclose(conditioning(a, 20, 10, a.f_max), conditioning(b, 20, 10, b.f_max))
    ra, rb = rasterize(a, (20, 30), a.f_max), rasterize(b, (20, 30), b.f_max)
    np.testing.assert_array_equal(ra, rb)
    assert ra[2].sum() > 0 and ra[5].sum() > 0
