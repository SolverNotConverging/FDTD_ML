"""Hard grading, global allocation, fixed collars, and optimization evidence."""

from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import linprog

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.mesh import (
    AxisCollar,
    AxisConstraints,
    Mesh,
    MeshInfeasibleError,
    adjacent_ratio,
    axis_mesh,
)
from fdtdmesh.mesh_projection import density_quantiles


def test_default_grading_recovers_previous_allocation_failure():
    x, stats = axis_mesh(0.02, 10, [1, 100], [0.01], return_diagnostics=True)
    assert 0.01 in x and len(x) == 11
    assert adjacent_ratio(np.diff(x)) <= 1.4 + 1e-8
    assert np.flatnonzero(x == 0.01)[0] > 1
    assert stats["projection_status"] == "optimal" and stats["mip_gap"] == 0
    assert stats["projection_l1"] > 0
    np.testing.assert_array_equal(x, axis_mesh(0.02, 10, [1, 100], [0.01]))


def test_milp_matches_exhaustive_anchor_assignment_optimum():
    n, anchors = 7, [0.29, 0.64]
    target = density_quantiles(1, n, [1, 5, 1])
    x, stats = axis_mesh(1, n, [1, 5, 1], anchors, return_diagnostics=True)
    # Independent LP per assignment: exhaustive global reference for this small case.
    costs = []
    for indices in combinations(range(1, n), len(anchors)):
        nv = (n + 1) + (n - 1)
        bounds = [(0, 1)] * nv
        bounds[0], bounds[n] = (0, 0), (1, 1)
        for index, a in zip(indices, anchors):
            bounds[index] = (a, a)
        rows, rhs = [], []
        for i in range(n):
            row = np.zeros(nv)
            row[i] = 1
            row[i + 1] = -1
            rows.append(row)
            rhs.append(-1e-9)
        for i in range(n - 1):
            for coefficients in ([1.4, -2.4, 1], [-1, 2.4, -1.4]):
                row = np.zeros(nv)
                row[i : i + 3] = coefficients
                rows.append(row)
                rhs.append(0)
        for i in range(1, n):
            for sign in (1, -1):
                row = np.zeros(nv)
                row[i] = sign
                row[n + i] = -1
                rows.append(row)
                rhs.append(sign * target[i])
        objective = np.r_[np.zeros(n + 1), np.ones(n - 1) / (n - 1)]
        result = linprog(objective, A_ub=rows, b_ub=rhs, bounds=bounds, method="highs")
        if result.success:
            costs.append(result.fun)
    assert costs
    assert stats["projection_l1"] == pytest.approx(min(costs), abs=1e-9)
    assert all(a in x for a in anchors)


def test_mandatory_grading_cannot_be_disabled_or_bypassed():
    for ratio in (None, 1.5, 0.9, np.inf):
        with pytest.raises(ValueError):
            AxisConstraints(max_ratio=ratio)
    with pytest.raises(MeshInfeasibleError):
        Mesh([0, 0.01, 1], [0, 0.5, 1])
    with pytest.raises(MeshInfeasibleError):
        axis_mesh(1, 2, [1], [0.01])


def test_fixed_collar_budget_spacing_and_interior_density_only():
    collar = AxisCollar(4, 0.1)
    x = axis_mesh(1, 30, [10, 2, 1, 3, 20], collar=collar)
    for i, value in collar.fixed_lines(1, 30).items():
        assert x[i] == value
    assert adjacent_ratio(np.diff(x)) <= 1.4 + 1e-8
    a = axis_mesh(1, 30, [1, 2, 3, 4, 5, 5, 4, 3, 2, 1], collar=collar)
    b = axis_mesh(1, 30, [1e100, 2, 3, 4, 5, 5, 4, 3, 2, 1e100], collar=collar)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_pml_fixed_when_budget_changes_and_scene_exclusion():
    s = FDTD_2D_Ez(0.02, 0.015, 80, 60, 20e9, Nt=3)
    s.add_PML(8)
    t = s.pml.x.thickness
    s.mesh_from_density([1], [1], Nx=100, Ny=80)
    assert s.mesh.x[8] == t and len(s.mesh.x) == 101
    s.add_circle("PEC", (0.001, 0.007), 0.0001)
    with pytest.raises(ValueError, match="Geometry"):
        s._prepare()


def test_grading_moves_lines_on_both_sides_of_close_anchors():
    target = density_quantiles(0.02, 40, [1])
    x = axis_mesh(0.02, 40, [1], [0.0098, 0.0102])
    assert 0.0098 in x and 0.0102 in x
    assert np.any(abs(x[target < 0.0098] - target[target < 0.0098]) > 1e-6)
    assert np.any(abs(x[target > 0.0102] - target[target > 0.0102]) > 1e-6)


def test_optimization_timeout_is_not_reported_as_infeasible(monkeypatch):
    from types import SimpleNamespace

    from fdtdmesh import MeshOptimizationError, mesh_projection

    monkeypatch.setattr(
        mesh_projection,
        "milp",
        lambda *a, **kw: SimpleNamespace(status=1, success=False, message="time limit"),
    )
    with pytest.raises(MeshOptimizationError, match="time limit"):
        axis_mesh(1, 10, [1, 100], [0.5])


def test_uniform_pml_roundoff_and_failed_remesh_preserves_previous_mesh():
    s = FDTD_2D_Ez(0.02, 0.015, 80, 60, 20e9, Nt=1)
    s.add_PML(6)
    original = s.mesh_uniform()
    s.pml.validate_mesh(original)
    with pytest.raises(ValueError, match="fixed PML"):
        s.mesh_uniform(Nx=82)
    assert s.mesh is original and s.Nx == 80
