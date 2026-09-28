"""Public workflow contracts and physical integration of the broadband source."""

import matplotlib
import numpy as np
import pytest
from legacy_fixtures import fixed_simulation

matplotlib.use("Agg")

from fdtdmesh import C0, Geometry, Result, Simulation, SolverSettings
from fdtdmesh.mesh import MeshInfeasibleError
from fdtdmesh.pulse import GaussianPulse
from fdtdmesh.scene import Scene2D
from fdtdmesh.simulation import Convergence as DFTConvergence


def cylinder(**kwargs):
    lam = C0 / 1e9
    settings = kwargs.pop("settings", kwargs.pop("solver", None))
    sim = fixed_simulation((6 * lam, 6 * lam), 0.9e9, 1.1e9, settings=settings, **kwargs)
    sim.add_circle((3 * lam, 3 * lam), 0.47 * lam)
    return sim


def test_exact_geometry_survives_mesh_and_archive(tmp_path):
    sim = cylinder()
    original = sim.geometry
    pixels = original.rasterize((51, 73), channels=("occupancy", "x", "y"))
    sim._apply_mesh("uniform", cells=(144, 144), strict=True)
    sim.save(tmp_path / "simulation.h5")
    restored = Simulation.load(tmp_path / "simulation.h5")
    assert restored.geometry == original
    np.testing.assert_array_equal(restored.mesh.x, sim.mesh.x)
    sim._apply_mesh("deterministic", cells=(144, 144))
    assert sim.geometry is original
    np.testing.assert_array_equal(
        pixels, sim.geometry.rasterize((51, 73), channels=("occupancy", "x", "y"))
    )
    assert sim.mesh.metadata["strategy"] == "deterministic"
    assert not sim.summary()["is_uniform"]
    original.save(tmp_path / "geometry.json")
    assert Geometry.load(tmp_path / "geometry.json") == original
    with pytest.raises(ValueError):
        sim.mesh.x.flags.writeable = True
    with pytest.raises(ValueError, match="material"):
        sim.add_circle((0.9, 0.9), 0.05, material="glass")
    assert sim.mesh is not None  # Failed edits are transactional.
    handle = sim.add_circle((0.9, 0.9), 0.05, material="air")
    assert sim.mesh is None
    with pytest.raises(RuntimeError, match="apply_mesh"):
        sim.solve()
    sim.remove_geometry(handle)
    assert sim.geometry.shapes == original.shapes


def test_continuous_overlays_merge_and_preserve_positive_gaps():
    a = Scene2D(10, 10).add_rectangle((2, 5), (2, 8)).add_rectangle((5, 8), (2, 8))
    b = Scene2D(10, 10).add_rectangle((2, 8), (2, 8))
    x, y = np.meshgrid(np.linspace(1, 9, 81), np.linspace(1, 9, 81))
    np.testing.assert_array_equal(a.contains(x, y), b.contains(x, y))
    a = Scene2D(10, 10).add_rectangle((1, 9), (1, 9))
    a.add_rectangle((3, 5), (3, 7), pec=False).add_rectangle((5, 7), (3, 7), pec=False)
    b = Scene2D(10, 10).add_rectangle((1, 9), (1, 9)).add_rectangle((3, 7), (3, 7), pec=False)
    np.testing.assert_array_equal(a.contains(x, y), b.contains(x, y))
    assert not a.contains(5, 5)
    assert a.contains(3, 5)  # Physical PEC/air wall remains PEC.
    for build in (
        lambda s, p: s.add_circle((5, 5), 2, pec=p),
        lambda s, p: s.add_rectangle((3, 7), (3, 7), pec=p),
    ):
        erased = build(build(Scene2D(10, 10), True), False)
        assert not erased.contains(x, y).any()
    gap = 1e-8
    a = (
        Scene2D(10, 10)
        .add_rectangle((2, 5 - gap / 2), (2, 8))
        .add_rectangle((5 + gap / 2, 8), (2, 8))
    )
    assert not a.contains(5, 5)
    # Curved tangent cuts retain the physical PEC cusp, not an artificial seam.
    a = Scene2D(10, 10).add_rectangle((1, 9), (1, 9))
    a.add_circle((4, 5), 1, pec=False).add_circle((6, 5), 1, pec=False)
    assert a.contains(5, 5)


def test_gaussian_band_and_sampled_source():
    pulse = GaussianPulse(0.9e9, 1.1e9)
    t = np.linspace(0, pulse.duration, 50001)
    f = np.array([pulse.fmin, pulse.frequency, pulse.fmax])
    sampled = abs(
        np.array([np.trapezoid(pulse(t) * np.exp(-2j * np.pi * freq * t), t) for freq in f])
    )
    np.testing.assert_allclose(sampled / sampled[1], pulse.spectrum(f), rtol=1e-10)
    np.testing.assert_allclose(20 * np.log10(sampled[[0, 2]] / sampled[1]), -6, atol=1e-8)
    assert pulse(-1) == 0 and pulse(pulse.duration + 1) == 0
    assert len(SolverSettings().frequencies(0.9e9, 1.1e9)) == 21
    with pytest.raises(ValueError):
        SolverSettings(dft_bins=(1e9, 0.9e9))
    with pytest.raises(ValueError):
        Simulation(fmin=1e9, fmax=1e9)


def test_mesh_policy_and_plotting(tmp_path):
    sim = cylinder()
    with pytest.raises(MeshInfeasibleError):
        sim._apply_mesh("uniform", cells=(140, 140), strict=True)
    sim._apply_mesh("uniform", cells=(144, 144), strict=True)
    mesh = sim.mesh
    with pytest.raises(ValueError):
        sim.apply_mesh("density", cells=(144, 144), density=([-1, 1], [1, 1]))
    assert sim.mesh is mesh
    import matplotlib.pyplot as plt

    for plot in (sim.plot_geometry, sim.plot_mesh, sim.plot_discretization, sim.plot_source):
        fig = plot()
        fig.savefig(tmp_path / (plot.__name__ + ".png"))
        plt.close(fig)
    sim.configure_solver(dft_bins=3)
    assert sim.mesh is None


@pytest.mark.cuda
def test_broadband_workflow_gpu_history_h5_and_cpu_plotting(tmp_path, monkeypatch):
    from fdtdmesh.scattering import cylinder_amplitude, pattern_error
    from fdtdmesh.solver import tmz

    try:
        tmz.cuda_backend()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    sim = cylinder(settings=SolverSettings(stop=DFTConvergence(check_interval=256)))
    sim._apply_mesh("uniform", cells=(144, 144), strict=True)
    result = sim.solve()
    assert result.converged and sim.result is result
    assert result.bin_history.shape == (len(result.history), 21, 3)
    np.testing.assert_allclose(
        np.max(result.bin_history[:, :, :2], axis=(1, 2)), result.history[:, 2]
    )
    assert result.diagnostics["stepping_field_transfers"] == 0
    assert result.diagnostics["debug_d2h_bytes"] == 0
    assert result.diagnostics["result_d2h_bytes"] == 21 * len(result.angles) * 3 * 8
    for i in (0, 10, 20):
        f = result.frequencies[i]
        exact = cylinder_amplitude(0.47 * C0 / 1e9, f, result.angles)
        assert pattern_error(result.width[i], 4 / (2 * np.pi * f / C0) * abs(exact) ** 2) < 0.03
    result.save(tmp_path / "result.h5")

    def unavailable():
        raise AssertionError("Loading/plotting must not initialize CUDA")

    monkeypatch.setattr(tmz, "cuda_backend", unavailable)
    loaded = Result.load(tmp_path / "result.h5")
    np.testing.assert_array_equal(loaded.far_field, result.far_field)
    np.testing.assert_array_equal(loaded.bin_history, result.bin_history)
    assert loaded.geometry == sim.geometry
    assert loaded.configuration == result.configuration
    with pytest.raises(ValueError):
        loaded.far_field.flags.writeable = True
    assert len(loaded.save_plots(tmp_path / "plots")) == 6


@pytest.mark.cuda
def test_unsuccessful_run_exposes_diagnostics_without_qualifying_result():
    from fdtdmesh import ConvergenceError
    from fdtdmesh.solver.tmz import cuda_backend

    try:
        cuda_backend()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    sim = cylinder(settings=SolverSettings(stop=DFTConvergence(max_steps=1700, check_interval=256)))
    sim._apply_mesh("uniform", cells=(144, 144), strict=True)
    with pytest.raises(ConvergenceError) as failure:
        sim.solve()
    assert not failure.value.result.converged and sim.result is None
    assert failure.value.result.diagnostics["status"] == "max_steps"
