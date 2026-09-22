"""Independent physical checks for the new solver; no legacy solver imports."""

import numpy as np
import pytest
from scipy.special import hankel1

from scattermesh import Circle, Grid, Material, PlaneWave, Rectangle, focused_axis, simulate
from scattermesh.analytic import cylinder_width
from scattermesh.constants import C0, MU0
from scattermesh.geometry import average_materials
from scattermesh.observables import Contour, SurfaceDFT


def test_grading_is_selectable_and_nodes_are_not_inserted():
    x = np.r_[0, np.cumsum([1, 1, 2.5, 1, 1])]
    grid = Grid(x, x, max_ratio=3)
    assert grid.shape == (6, 6)
    assert grid.diagnostics()["x_grading"] == 2.5
    with pytest.raises(ValueError, match="grading"):
        Grid(x, x, max_ratio=2)
    Grid(x, x, max_ratio=None)
    with pytest.raises(ValueError):
        Grid([0, 1, 1, 2, 3], x)


def test_sampled_area_and_material_overlap():
    x = focused_axis(1, 48, strength=2)
    grid = Grid(x, x)
    circle = Circle((0.5, 0.5), 0.15, Material(5, 0.02))
    eps, sigma = average_materials(grid, [circle], samples=16)
    bounds = np.r_[0, (x[:-1] + x[1:]) / 2, 1]
    area = np.sum((eps - 1) / 4 * np.outer(np.diff(bounds), np.diff(bounds)))
    assert area == pytest.approx(np.pi * 0.15**2, rel=0.002)
    np.testing.assert_allclose(sigma, 0.02 * (eps - 1) / 4, atol=1e-15)
    # Last geometry wins per quadrature sample, including vacuum cut-outs.
    hole = Rectangle((0.4, 0.6, 0.4, 0.6), Material())
    eps, sigma = average_materials(grid, [circle, hole], samples=8)
    assert eps[24, 24] == 1
    assert sigma[24, 24] == 0


def test_plane_wave_translation_and_angle():
    wave = PlaneWave(1e9, 1e-9, 9e-9, angle=0.71)
    distance = 0.123
    t = np.linspace(8e-9, 10e-9, 100)
    np.testing.assert_allclose(
        wave.electric(distance * np.cos(0.71), distance * np.sin(0.71), t + distance / C0),
        wave.pulse(t),
        atol=1e-14,
    )


def test_nf2ff_complex_phase_and_normalization_against_hankel_wave():
    x = np.linspace(0, 1, 201)
    contour = Contour(Grid(x, x), (0.2, 0.8, 0.2, 0.8))
    monitor = SurfaceDFT(contour, [1e9])
    center = np.array([0.47, 0.54])
    p = contour.points - center
    r = np.linalg.norm(p, axis=1)
    omega, k = 2 * np.pi * 1e9, 2 * np.pi * 1e9 / C0
    monitor.electric[0] = hankel1(0, k * r)
    gradient = (-k * hankel1(1, k * r))[:, None] * p / r[:, None]
    hx, hy = -1j * gradient[:, 1] / (omega * MU0), 1j * gradient[:, 0] / (omega * MU0)
    monitor.tangential_h[0] = -contour.normals[:, 1] * hx + contour.normals[:, 0] * hy
    monitor.incident[:] = 1
    angles = np.linspace(0, 2 * np.pi, 91, endpoint=False)
    phase = np.exp(-1j * k * (center[0] * np.cos(angles) + center[1] * np.sin(angles)))
    expected = np.sqrt(2 / (np.pi * k)) * np.exp(-1j * np.pi / 4) * phase
    np.testing.assert_allclose(monitor.far_amplitude(angles)[0], expected, rtol=3e-4)
    np.testing.assert_allclose(monitor.scattering_width(angles)[0], 4 / k, rtol=6e-4)


def test_surface_dft_retains_e_h_time_stagger():
    grid = Grid(np.linspace(0, 1, 9), np.linspace(0, 1, 9))
    monitor = SurfaceDFT(Contour(grid, (0.25, 0.75, 0.25, 0.75)), [1.0])
    dt, steps = 0.01, 100
    for n in range(steps):
        te, th = (n + 1) * dt, (n + 0.5) * dt
        monitor.accumulate(
            np.full(grid.shape, np.cos(2 * np.pi * te + 0.3)),
            np.full((9, 8), 2 * np.cos(2 * np.pi * th - 0.2)),
            np.full((8, 9), 3 * np.cos(2 * np.pi * th - 0.2)),
            electric_time=te,
            magnetic_time=th,
            dt=dt,
            incident=np.cos(2 * np.pi * te),
        )
    np.testing.assert_allclose(monitor.electric, 0.5 * np.exp(-0.3j), atol=1e-14)
    tangent = -2 * monitor.contour.normals[:, 1] + 3 * monitor.contour.normals[:, 0]
    np.testing.assert_allclose(monitor.tangential_h[0], 0.5 * tangent * np.exp(0.2j), atol=1e-14)
    np.testing.assert_allclose(monitor.incident, 0.5, atol=1e-14)


@pytest.mark.parametrize("angle", [0, 0.71, 2.37])
def test_empty_domain_has_zero_scattering_on_graded_grid(angle):
    x = focused_axis(1.2, 32, strength=2)
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=angle, origin=(0.6, 0.6))
    result = simulate(Grid(x, x), [], source, frequencies=[1e9], duration=16e-9, pml_thickness=0.12)
    for field in result.fields.values():
        assert np.all(field == 0)
    assert np.all(result.monitor.scattering_width([0, 1]) == 0)


@pytest.mark.parametrize("conductivity", [0, 0.02])
def test_dielectric_cylinder_refines_toward_analytic_scattering(conductivity):
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    material = Material(4, conductivity)
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.53, origin=(0.6, 0.6))
    frequencies = [0.8e9, 1e9, 1.2e9]
    reference = np.array(
        [cylinder_width(0.06, material, f, angles, source.angle) for f in frequencies]
    )
    errors = []
    for cells in [48, 96]:
        x = focused_axis(1.2, cells, width=0.12, strength=2)
        result = simulate(
            Grid(x, x),
            [Circle((0.6, 0.6), 0.06, material)],
            source,
            frequencies=frequencies,
            duration=40e-9,
            pml_thickness=0.15,
            samples=12,
        )
        width = result.monitor.scattering_width(angles)
        errors.append(np.linalg.norm(width - reference, axis=1) / np.linalg.norm(reference, axis=1))
        assert result.diagnostics["tail_peak_over_global_peak"] < 1e-5
    assert np.all(errors[1] < errors[0])
    assert max(errors[1]) < 0.025


def test_observation_guards_reject_unreliable_setups():
    x = np.linspace(0, 1.2, 33)
    grid = Grid(x, x)
    source = PlaneWave(1e9, 1e-9, 9e-9, origin=(0.6, 0.6))
    with pytest.raises(ValueError, match="PML"):
        simulate(
            grid,
            [],
            source,
            frequencies=[1e9],
            duration=20e-9,
            pml_thickness=0.25,
            monitor_bounds=(0.2, 1.0, 0.2, 1.0),
        )
    with pytest.raises(ValueError, match="buffer"):
        simulate(
            grid,
            [Circle((0.3, 0.6), 0.1, Material(4))],
            source,
            frequencies=[1e9],
            duration=20e-9,
            pml_thickness=0.12,
        )
    with pytest.raises(ValueError, match="delay"):
        simulate(
            grid,
            [],
            PlaneWave(1e9, 1e-9, 1e-9),
            frequencies=[1e9],
            duration=20e-9,
            pml_thickness=0.12,
        )
    result = simulate(grid, [], source, frequencies=[5e9], duration=20e-9, pml_thickness=0.12)
    with pytest.raises(ValueError, match="spectrum"):
        result.monitor.scattering_width([0])


def test_nonempty_scene_with_abrupt_grading_ratio_2_8():
    # Unlike vacuum, this exercises propagating scattered fields across jumps.
    widths = np.tile([1.0, 2.8], 32)
    x = np.r_[0, np.cumsum(1.2 * widths / widths.sum())]
    material = Material(4)
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    result = simulate(
        Grid(x, x),
        [Circle((0.6, 0.6), 0.06, material)],
        source,
        frequencies=[1e9],
        duration=40e-9,
        pml_thickness=0.15,
        samples=16,
    )
    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    reference = cylinder_width(0.06, material, 1e9, angles, source.angle)
    error = np.linalg.norm(result.monitor.scattering_width(angles)[0] - reference) / np.linalg.norm(
        reference
    )
    assert error < 0.06
    assert result.diagnostics["tail_peak_over_global_peak"] < 1e-5
