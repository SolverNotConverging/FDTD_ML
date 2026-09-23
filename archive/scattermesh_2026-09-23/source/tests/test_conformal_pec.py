"""Off-grid PEC physics and a separate stability-bound check."""

import json

import numpy as np
import pytest

from scattermesh import PEC, Circle, Grid, Material, PlaneWave, Rectangle, focused_axis, simulate
from scattermesh.analytic import cylinder_width
from scattermesh.conformal import CutCellPEC
from scattermesh.constants import C0
from scattermesh.geometry import average_materials


def test_exact_rectangle_cuts_preserve_nodes_and_restrict_time_step():
    x = np.linspace(0, 1, 21)
    grid = Grid(x, x)
    box = Rectangle((0.30005, 0.6, 0.321, 0.581), PEC())
    cut = CutCellPEC(grid, [box])
    np.testing.assert_array_equal(grid.x, x)
    # The exterior node at x=.3 is only .00005 m from the PEC boundary.
    assert cut.minimum_fraction == pytest.approx(0.001, rel=1e-9)
    assert cut.inverse_lengths[0][6, 8] == pytest.approx(1 / 0.00005)
    assert cut.stable_time_step() < 0.1 * 0.05 / (C0 * np.sqrt(2))
    with pytest.raises(ValueError, match="averaging"):
        average_materials(grid, [box])


def test_cut_cell_cfl_bound_against_independent_matrix_eigenvalues():
    grid = Grid(focused_axis(1, 12, strength=1), focused_axis(1, 13, strength=2))
    pec = CutCellPEC(grid, [Rectangle((0.391, 0.61, 0.397, 0.62), PEC())])
    nodes = [
        (i, j)
        for i in range(1, len(grid.x) - 1)
        for j in range(1, len(grid.y) - 1)
        if not pec.inside[i, j]
    ]
    index = {p: i for i, p in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)))
    weights = np.zeros(len(nodes))
    ix, iy = pec.inverse_lengths
    for row, (i, j) in enumerate(nodes):
        dualx = (grid.x[i + 1] - grid.x[i - 1]) / 2
        dualy = (grid.y[j + 1] - grid.y[j - 1]) / 2
        weights[row] = dualx * dualy
        for neighbor, coefficient in [
            ((i - 1, j), ix[i - 1, j] / dualx),
            ((i + 1, j), ix[i, j] / dualx),
            ((i, j - 1), iy[i, j - 1] / dualy),
            ((i, j + 1), iy[i, j] / dualy),
        ]:
            matrix[row, row] += coefficient
            if neighbor in index:
                matrix[row, index[neighbor]] -= coefficient
    symmetric = np.sqrt(weights[:, None]) * matrix / np.sqrt(weights[None, :])
    np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    assert eigenvalues.min() > 0
    exact_limit = 2 / (C0 * np.sqrt(eigenvalues.max()))
    assert pec.stable_time_step() <= exact_limit


@pytest.mark.parametrize("strength", [0, 2])
def test_off_grid_pec_circle_accuracy_beats_staircase(strength):
    x = focused_axis(1.2, 64, width=0.12, strength=strength)
    grid = Grid(x, x)
    objects = [Circle((0.613, 0.591), 0.08, PEC())]
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    reference = cylinder_width(0.08, PEC(), 1e9, angles, source.angle)
    errors = {}
    for mode in ["staircase", "conformal"]:
        result = simulate(
            grid,
            objects,
            source,
            frequencies=[1e9],
            duration=35e-9,
            pml_thickness=0.15,
            pec_mode=mode,
        )
        json.dumps(result.diagnostics)
        errors[mode] = np.linalg.norm(
            result.monitor.scattering_width(angles)[0] - reference
        ) / np.linalg.norm(reference)
        assert result.diagnostics["tail_peak_over_global_peak"] < 1e-5
        if mode == "conformal":
            assert result.diagnostics["pec_subcell_edge_count"] > 0
            assert result.diagnostics["dt_fraction_of_grid_cfl"] < 1
        mask = objects[0].contains(grid.x[:, None], grid.y[None, :])
        total = result.fields["Ez"] + source.electric(
            grid.x[:, None], grid.y[None, :], result.diagnostics["simulated_time"]
        )
        np.testing.assert_allclose(total[mask], 0.0, atol=1e-15)
    assert errors["conformal"] < 0.04
    assert errors["conformal"] < 0.4 * errors["staircase"]


def test_unresolved_or_unsupported_pec_is_rejected_instead_of_disappearing():
    x = np.linspace(0, 1, 11)
    grid = Grid(x, x)
    with pytest.raises(ValueError, match="represented"):
        CutCellPEC(grid, [Circle((0.45, 0.45), 0.01, PEC())])
    # A small vacuum gap whose two surrounding PEC nodes belong to different boxes.
    with pytest.raises(ValueError, match="gap"):
        CutCellPEC(
            grid,
            [
                Rectangle((0.31, 0.441, 0.31, 0.69), PEC()),
                Rectangle((0.459, 0.69, 0.31, 0.69), PEC()),
            ],
        )
    wave = PlaneWave(1e9, 1e-9, 9e-9)
    with pytest.raises(ValueError, match="Overlapping or touching"):
        simulate(
            grid,
            [Rectangle((0.31, 0.6, 0.31, 0.6), PEC()), Circle((0.5, 0.5), 0.1, Material(4))],
            wave,
            frequencies=[1e9],
            duration=10e-9,
            pml_thickness=0.1,
        )


def test_enlargement_rejects_dielectric_loaded_transfer_stencil():
    axis = np.linspace(0, 1.2, 25)
    grid = Grid(axis, axis)
    objects = [
        Rectangle((0.50005, 0.7, 0.45, 0.65), PEC()),
        Rectangle((0.351, 0.499, 0.45, 0.65), Material(4, 0.01)),
    ]
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    with pytest.raises(ValueError, match="transfer stencil"):
        simulate(
            grid,
            objects,
            source,
            frequencies=[1e9],
            duration=12e-9,
            pml_thickness=0.15,
            pec_mode="enlarged",
        )


def test_enlargement_projects_mass_and_stiffness_consistently():
    grid = Grid(focused_axis(1, 13, strength=1), focused_axis(1, 14, strength=2))
    objects = [Circle((0.491, 0.509), 0.16, PEC())]
    cut = CutCellPEC(grid, objects, "enlarged")
    aggregate = cut.enlargement
    assert len(aggregate.slaves) > 0
    nodes = [
        (i, j)
        for i in range(1, len(grid.x) - 1)
        for j in range(1, len(grid.y) - 1)
        if not cut.inside[i, j]
    ]
    index = {p: i for i, p in enumerate(nodes)}
    stiffness = np.zeros((len(nodes), len(nodes)))
    mass = np.zeros(len(nodes))
    for row, (i, j) in enumerate(nodes):
        dualx, dualy = (grid.x[i + 1] - grid.x[i - 1]) / 2, (grid.y[j + 1] - grid.y[j - 1]) / 2
        mass[row] = dualx * dualy
        ix, iy = cut.inverse_lengths
        for neighbor, coefficient in [
            ((i - 1, j), ix[i - 1, j] * dualy),
            ((i + 1, j), ix[i, j] * dualy),
            ((i, j - 1), iy[i, j - 1] * dualx),
            ((i, j + 1), iy[i, j] * dualx),
        ]:
            stiffness[row, row] += coefficient
            if neighbor in index:
                stiffness[row, index[neighbor]] -= coefficient
    projection = np.zeros((len(nodes), len(aggregate.masters)))
    column = {master: c for c, master in enumerate(aggregate.masters)}
    constrained = {
        slave: (root, weight)
        for slave, root, weight in zip(aggregate.slaves, aggregate.roots, aggregate.slave_weights)
    }
    for row, point in enumerate(nodes):
        flat = np.ravel_multi_index(point, grid.shape)
        root, weight = constrained.get(flat, (flat, 1.0))
        projection[row, column[root]] = weight
    expected_mass = projection.T @ (mass[:, None] * projection)
    np.testing.assert_allclose(expected_mass, np.diag(aggregate.mass), atol=1e-14)
    expected_stiffness = projection.T @ stiffness @ projection
    np.testing.assert_allclose(expected_stiffness, aggregate.stiffness.toarray(), atol=1e-12)
    symmetric = expected_stiffness / np.sqrt(aggregate.mass[:, None] * aggregate.mass[None, :])
    eigenvalues = np.linalg.eigvalsh(symmetric)
    assert eigenvalues.min() > 0
    assert aggregate.stable_dt <= 2 / (C0 * np.sqrt(eigenvalues.max()))


def test_enlargement_removes_tiny_cut_penalty_without_moving_pec():
    x = focused_axis(1.2, 48)
    grid = Grid(x, x)
    rectangle = Rectangle((0.475 + 0.025 * 1e-6, 0.71025, 0.4873, 0.6891), PEC())
    plain, enlarged = [CutCellPEC(grid, [rectangle], mode) for mode in ("conformal", "enlarged")]
    for raw, merged in zip(plain.boundaries, enlarged.boundaries):
        np.testing.assert_array_equal(raw[4], merged[4])
        np.testing.assert_array_equal(raw[5], merged[5])
    assert plain.minimum_fraction == pytest.approx(1e-6, rel=1e-7)
    grid_dt = 0.025 / (C0 * np.sqrt(2))
    assert plain.stable_time_step() < 0.003 * grid_dt
    assert enlarged.stable_time_step() >= 0.999 * grid_dt
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    result = simulate(
        grid,
        [rectangle],
        source,
        frequencies=[1e9],
        duration=40e-9,
        pml_thickness=0.15,
        pec_mode="enlarged",
    )
    assert result.diagnostics["dt_fraction_of_grid_cfl"] == pytest.approx(1)
    assert result.diagnostics["tail_peak_over_global_peak"] < 1e-5


def test_enlarged_scattering_accuracy_and_total_field_constraint():
    x = focused_axis(1.2, 64, width=0.12, strength=2)
    grid = Grid(x, x)
    circle = Circle((0.613, 0.591), 0.08, PEC())
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    result = simulate(
        grid,
        [circle],
        source,
        frequencies=[1e9],
        duration=35e-9,
        pml_thickness=0.15,
        pec_mode="enlarged",
    )
    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    reference = cylinder_width(0.08, PEC(), 1e9, angles, source.angle)
    error = np.linalg.norm(result.monitor.scattering_width(angles)[0] - reference) / np.linalg.norm(
        reference
    )
    assert error < 0.03
    assert result.diagnostics["dt_fraction_of_grid_cfl"] == pytest.approx(1)
    # Probe while the source is active, so the affine incident-field correction
    # cannot accidentally pass merely because the final pulse has decayed.
    result = simulate(
        grid,
        [circle],
        source,
        frequencies=[1e9],
        duration=9.3e-9,
        pml_thickness=0.15,
        pec_mode="enlarged",
    )
    aggregation = CutCellPEC(grid, [circle], "enlarged").enlargement
    total = result.fields["Ez"] + source.electric(
        grid.x[:, None], grid.y[None, :], result.diagnostics["simulated_time"]
    )
    np.testing.assert_allclose(
        total.ravel()[aggregation.slaves],
        aggregation.slave_weights * total.ravel()[aggregation.roots],
        atol=1e-14,
    )
