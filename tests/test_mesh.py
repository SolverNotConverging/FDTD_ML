import numpy as np
import pytest

from fdtdmesh.mesh import AxisConstraints, Mesh, MeshInfeasibleError, axis_mesh
from fdtdmesh.mesh_projection import density_quantiles


@pytest.mark.parametrize("n", [1, 2, 7, 31, 100])
def test_uniform_exact_count(n):
    x = axis_mesh(0.02, n, np.ones(43))
    np.testing.assert_allclose(x, np.linspace(0, 0.02, n + 1), atol=1e-16)
    assert len(x) == n + 1 and x[0] == 0 and x[-1] == 0.02


def test_anchors_density_and_repeatability():
    rho = np.r_[np.ones(50), np.ones(50) * 8]
    x = axis_mesh(1, 40, rho, [0.17, 0.61, 0.17])
    assert 0.17 in x and 0.61 in x and len(x) == 41
    assert np.sum(x > 0.5) > np.sum(x < 0.5)
    np.testing.assert_array_equal(x, axis_mesh(1, 40, rho, [0.61, 0.17]))


def test_quantiles_piecewise_constant():
    # Masses 1/2 and 3/2, so half the four cells occupy the final third.
    np.testing.assert_allclose(density_quantiles(1, 4, [1, 3]), [0, 0.5, 2 / 3, 5 / 6, 1])


def test_projection_preserves_anchors_and_bounds():
    x = axis_mesh(
        0.01,
        32,
        [1, 1, 10, 2, 1],
        [0.003, 0.007],
        AxisConstraints(min_spacing=0.00015, max_spacing=0.0006, max_ratio=1.4),
    )
    h = np.diff(x)
    assert len(x) == 33 and 0.003 in x and 0.007 in x
    assert h.min() >= 0.00015 - 1e-12 and h.max() <= 0.0006 + 1e-12
    assert max((h[1:] / h[:-1]).max(), (h[:-1] / h[1:]).max()) <= 1.4 + 1e-8


@pytest.mark.parametrize(
    "args",
    [
        (1, 2, [1], [0.2, 0.4]),
        (1, 10, [1], [0.01], AxisConstraints(min_spacing=0.02)),
        (1, 2, [1], [], AxisConstraints(max_spacing=0.2)),
    ],
)
def test_infeasible(args):
    with pytest.raises(MeshInfeasibleError):
        axis_mesh(*args)


@pytest.mark.parametrize("rho", [[0, 1], [-1, 2], [np.nan], [np.inf], []])
def test_invalid_density(rho):
    with pytest.raises(ValueError):
        axis_mesh(1, 10, rho)


def test_randomized_invariants():
    rng = np.random.default_rng(83)
    for _ in range(20):
        count = int(rng.integers(8, 40))
        # Anchors drawn from a known graded witness; arbitrary close anchors can
        # legitimately be infeasible now that grading is mandatory.
        widths = 1 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, count))
        witness = np.r_[0, np.cumsum(widths) / widths.sum()]
        anchors = rng.choice(witness[1:-1], 5, replace=False)
        x = axis_mesh(1, count, np.exp(rng.normal(size=32)), anchors)
        assert len(x) == count + 1 and np.all(np.diff(x) > 0)
        assert all(a in x for a in anchors)


def test_mesh_owns_readonly_data():
    x = np.linspace(0, 1, 5)
    mesh = Mesh(x, x)
    x[1] = 0.3
    assert mesh.x[1] == 0.25 and not mesh.x.flags.writeable
