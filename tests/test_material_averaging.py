from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.constants import EPS0
from fdtdmesh.evaluation import EvaluationConfig
from fdtdmesh.solver.averaging import _sample_pairs, average_electric
from fdtdmesh.solver.coefficients import build_coefficients
from fdtdmesh.solver.reference_tmz import run_reference
from fdtdmesh.solver.tmz import cuda_available, cuda_backend


def scene():
    s = FDTD_2D_Ez(1, 1, 16, 16, 1e9, Nt=30, dtype="float64")
    s.add_material("a", epsilon_r=12, sigma_e=0.4)
    s.add_rectangle("a", (0.5, 1), (0, 1))
    s.mesh_uniform()
    return s


def test_exact_half_fraction_and_precoefficient_loss_averaging():
    s = scene()
    old = build_coefficients(s, s.mesh, dtype="float64")
    s.material_averaging = "sampled"
    new = build_coefficients(s, s.mesh, dtype="float64")
    i, j = 8, 8
    # epsilon=6.5, sigma=.2, not a mean of the nonlinear update coefficients.
    loss = 0.2 * new.dt / (2 * EPS0 * 6.5)
    assert new.ca[i, j] == pytest.approx((1 - loss) / (1 + loss))
    assert new.current_scale[i, j] == pytest.approx(new.dt / (EPS0 * 6.5) / (1 + loss))
    np.testing.assert_array_equal(old.pec, new.pec)
    np.testing.assert_array_equal(old.chx, new.chx)
    assert old.dt == new.dt


def test_nonuniform_dual_cell_fraction_and_overlap_priority():
    s = scene()
    s.add_material("b", epsilon_r=20, sigma_e=0.8)
    s.add_rectangle("b", (0.55, 1), (0, 1))
    mesh = SimpleNamespace(x=np.array([0, 0.4, 0.6, 1]), y=np.array([0, 0.4, 0.6, 1]))
    eps, _, sig, pec = s.sample(mesh.x, mesh.y)
    e, c, d = average_electric(s, mesh, eps, sig, pec, samples=12, max_samples=12)
    # Cell at x=.6 spans [.5,.8]: 1/6 material a, 5/6 material b.
    assert e[2, 1] == pytest.approx((12 + 5 * 20) / 6)
    assert c[2, 1] == pytest.approx((0.4 + 5 * 0.8) / 6)
    assert not d["sampling_convergence_checked"]


def test_paired_sampling_matches_authoritative_geometry():
    s = scene()
    s.add_circle("vacuum", (0.65, 0.6), 0.1)
    s.add_triangle("a", [(0.1, 0.1), (0.4, 0.2), (0.3, 0.7)])
    s.add_rectangle("PEC", (0.7, 0.9), (0.7, 0.9))
    rng = np.random.default_rng(9)
    x, y = rng.uniform(0, 1, (2, 100))
    eps, sig, pec = _sample_pairs(s, x, y)
    for k in range(len(x)):
        expected = s.sample([x[k]], [y[k]])
        assert eps[k] == expected[0][0, 0]
        assert sig[k] == expected[2][0, 0]
        assert pec[k] == expected[3][0, 0]


def test_pec_interfaces_and_lines_remain_unsmoothed():
    s = scene()
    s.add_circle("PEC", (0.5, 0.5), 0.17)
    s.add_pec_line(x=0.5, y=(0.1, 0.3))
    eps, _, sig, pec = s.sample(s.mesh.x, s.mesh.y)
    e, c, d = average_electric(s, s.mesh, eps, sig, pec)
    # Interface at x=.5 would otherwise average. It crosses PEC and a thin line.
    for j in (3, 8, 10):
        assert e[8, j] == eps[8, j] and c[8, j] == sig[8, j]
    assert d["unexpected_pec_fallback_cells"] == 0


def test_tiny_enclosed_geometry_not_missed_by_homogeneous_corners():
    s = FDTD_2D_Ez(1, 1, 8, 8, 1e9, Nt=1)
    s.add_material("a", epsilon_r=30)
    s.add_circle("a", (0.51, 0.51), 0.025)
    s.mesh_uniform()
    eps, _, sig, pec = s.sample(s.mesh.x, s.mesh.y)
    e, _, d = average_electric(s, s.mesh, eps, sig, pec, samples=16, max_samples=32)
    assert 1 < e[4, 4] < 30
    assert d["candidate_cells"] > 0


def test_sampling_configuration_and_magnetic_scope():
    for values in [
        dict(material_averaging="blur"),
        dict(averaging_samples=0),
        dict(averaging_max_samples=17),
        dict(averaging_tolerance=float("nan")),
    ]:
        with pytest.raises(ValueError):
            replace(EvaluationConfig(), **values)
    s = scene()
    s.add_material("magnetic", mu_r=2)
    s.material_averaging = "sampled"
    with pytest.raises(ValueError, match="mu_r=1"):
        build_coefficients(s, s.mesh)


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_sampled_lossy_coefficients_cuda_numpy():
    s = scene()
    s.material_averaging = "sampled"
    s.add_source("point", x=0.25, y=0.5, width=1e-9, delay=4e-9)
    s.add_receiver("point", x=0.75, y=0.5)
    c, fields, sites, waves, receivers, *_ = s._prepare()
    expected = run_reference(c, fields, sites, waves, receivers)
    actual = cuda_backend().run(c, fields, sites, waves, receivers)
    for a, b in zip(actual[:4], expected):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-13)


def test_overwritten_pec_body_keeps_conservative_point_treatment():
    s = FDTD_2D_Ez(1, 1, 16, 16, 1e9, Nt=1)
    s.add_rectangle("PEC", (0.2, 0.8), (0.2, 0.8))
    s.add_material("a", epsilon_r=12)
    s.add_rectangle("a", (0.5, 0.9), (0.3, 0.7))
    s.mesh_uniform()
    eps, _, sig, pec = s.sample(s.mesh.x, s.mesh.y)
    averaged, _, diagnostics = average_electric(s, s.mesh, eps, sig, pec)
    assert averaged[8, 8] == eps[8, 8] == 12
    assert diagnostics["unexpected_pec_fallback_cells"] == 0


def test_reference_identity_distinguishes_mapping_and_preserves_legacy_point():
    from fdtdmesh.evaluation.references import _run_identity

    def identity(config):
        return _run_identity({"dataset_id": "test"}, config, ("train",), None, None, ())

    config = EvaluationConfig()
    point = identity(config)
    assert "material_averaging" not in point["config"]
    assert point == identity(replace(config, averaging_max_samples=64))
    sampled = identity(replace(config, material_averaging="sampled"))
    assert sampled["config"]["material_averaging"] == "sampled"
    assert sampled != point
