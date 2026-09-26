import numpy as np
import pytest
from scipy.linalg import eigvalsh

from fdtdmesh.mesh import Mesh
from fdtdmesh.scene import Scene2D
from fdtdmesh.solver.coefficients import build_coefficients
from fdtdmesh.solver.conformal import (
    UnresolvedGeometryError,
    apply_operator,
    build_conformal,
    enlarge,
    stiffness,
)


def test_enlarged_pair_conserves_circulation_and_is_positive():
    length = np.array([[1e-10], [1.0], [1.0]])
    mask = np.array([[True], [False], [False], [False]])
    op = enlarge(length, np.ones_like(length), mask, 0)
    f = np.array([[0.7], [-0.3], [0.2]])
    g = apply_operator(op, f)
    assert np.sum(length * g) == pytest.approx(np.sum(f), abs=1e-14)
    q = np.diag(op.diagonal.ravel())
    for i in range(3):
        if op.peer[i, 0] >= 0:
            q[i, op.peer[i, 0]] = op.coupling[i, 0]
    assert np.allclose(q, q.T)
    assert eigvalsh(q).min() > 0
    assert q.max() < 3  # No inverse-tiny-cut coefficient.


def test_cut_geometry_and_spectral_bound():
    mesh = Mesh(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    scene = Scene2D(1, 1)
    scene.add_circle((0.5, 0.5), 0.213)
    pec, hx, hy, bound = build_conformal(scene, mesh)
    # The horizontal centre line has exact intersections at .287 and .713.
    assert hy.length[5, 10] == pytest.approx(0.037)
    assert hy.length[14, 10] == pytest.approx(0.037)
    assert len(hx.pairs) + len(hy.pairs) > 0
    K, mass, _ = stiffness(mesh, pec, hx, hy)
    A = K.toarray() / np.sqrt(mass[:, None] * mass[None, :])
    assert np.allclose(A, A.T, atol=1e-12)
    eig = eigvalsh(A)
    assert eig.min() > 0
    assert eig.max() <= bound * (1 + 1e-12)
    c = build_coefficients(scene, mesh)
    assert c.dt <= 0.9 * c.dt_cartesian
    with pytest.raises(ValueError, match="below conformal bound"):
        build_coefficients(scene, mesh, dt=c.dt_bound * 1.001)


def test_subcell_topology_is_rejected_instead_of_silently_erased():
    mesh = Mesh(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    scene = Scene2D(1, 1)
    scene.add_circle((0.45, 0.5), 0.02)
    with pytest.raises(UnresolvedGeometryError, match="two vacuum endpoints"):
        build_conformal(scene, mesh)
    hidden = Scene2D(1, 1).add_circle((0.45, 0.45), 0.02)
    with pytest.raises(UnresolvedGeometryError, match="entirely inside"):
        build_conformal(hidden, mesh)


def test_vacuum_overlay_and_invalid_polygon():
    scene = Scene2D(1, 1)
    scene.add_rectangle((0.2, 0.8), (0.2, 0.8))
    scene.add_rectangle((0.4, 0.6), (0.4, 0.9), pec=False)
    assert scene.contains(0.3, 0.5)
    assert not scene.contains(0.5, 0.5)
    assert scene.contains(0.4, 0.5)  # A vacuum hole retains its PEC wall.
    assert scene.vacuum_intervals(0, 0.5) == pytest.approx([(0, 0.2), (0.4, 0.6), (0.8, 1)])
    with pytest.raises(ValueError):
        scene.add_polygon([(0.1, 0.1), (0.8, 0.8), (0.1, 0.8), (0.8, 0.1)])
