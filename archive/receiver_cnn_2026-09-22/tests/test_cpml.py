import numpy as np
import pytest

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.sampling import dual_areas, point_stencil
from fdtdmesh.solver.reference_tmz import run_reference
from fdtdmesh.solver.tmz import cuda_available, cuda_backend

cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("direction", ["x", "y", "xy"])
def test_cpml_cuda_numpy_parity(dtype, direction):
    s = FDTD_2D_Ez(0.024, 0.020, 64, 56, 50e9, Nt=500, dtype=dtype)
    s.add_PML(8, thickness=0.003, direction=direction)
    mat = s.add_material("loss", epsilon_r=3, mu_r=1.3, sigma_e=0.02)
    s.add_circle(mat, (0.014, 0.01), 0.0015)
    s.add_source(
        "point",
        x=0.0091,
        y=0.0101,
        width=1e-11,
        delay=4e-11,
        normalization="current",
        amplitude=0.001,
    )
    s.add_receiver("line", x=0.016, y=(0.008, 0.012), samples=5)
    s.mesh_from_density([1, 1.5, 2, 1], [1, 2, 1])
    c, f, sites, w, rx, *_ = s._prepare()
    cpu = run_reference(c, f, sites, w, rx)
    gpu = cuda_backend().run(c, f, sites, w, rx)
    for actual, expected in zip(gpu[:4], cpu):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=3e-5 if dtype == "float32" else 1e-11,
            atol=2e-5 if dtype == "float32" else 1e-12,
        )
    assert gpu[4]["stepping_transfers"] == 0
    assert np.max(abs(gpu[3])) > 1e-3
    assert np.all(gpu[0][c.pec.astype(bool)] == 0)


def test_profiles_are_staggered_and_identity_in_interior():
    s = FDTD_2D_Ez(0.02, 0.02, 40, 40, 20e9, Nt=1)
    s.add_PML(6, thickness=0.003)
    s.mesh_from_density([1], [1])
    c, *_ = s._prepare()
    ex, ey, hx, hy = np.split(c.cpml, [41, 82, 122])
    np.testing.assert_array_equal(ex[6:-6], np.tile([1, 1, 0], (29, 1)))
    assert ex[0, 0] < hx[0, 0] < 1
    assert np.all(c.cpml[:, 1] <= 1) and np.all(c.cpml[:, 1] > 0)
    assert np.all(c.cpml[:, 2] <= 0)


@pytest.mark.parametrize("kind", ["circle", "source", "receiver"])
def test_pml_scene_exclusion(kind):
    s = FDTD_2D_Ez(0.02, 0.02, 40, 40, 20e9, Nt=1)
    s.add_PML(6)
    if kind == "circle":
        s.add_circle("PEC", (0.001, 0.01), 0.0001)
    elif kind == "source":
        s.add_source("point", x=0.001, y=0.01)
    else:
        s.add_receiver("point", x=0.001, y=0.01)
    with pytest.raises(ValueError, match="inside"):
        s.mesh_from_density([1], [1])


def test_current_source_total_is_independent_of_grid_and_dt():
    for n, dt in ((20, 1e-12), (32, 5e-13)):
        s = FDTD_2D_Ez(0.02, 0.02, n, n, 20e9, Nt=5, dt=dt, dtype="float64")
        s.add_source(
            "point",
            x=0.00813,
            y=0.00937,
            amplitude=0.002,
            normalization="current",
            waveform="gaussian",
            delay=0,
            width=1e-8,
        )
        c, _, sites, w, *_ = s._prepare()
        reconstructed = (-w / c.current_scale.ravel()[sites]) * dual_areas(s.mesh).ravel()[sites]
        expected = s.sources[0][1].sample((np.arange(5) + 0.5) * dt, s.f_max)
        np.testing.assert_allclose(reconstructed.sum(axis=1), expected, rtol=1e-13)


def test_receiver_interpolation_exact_for_bilinear_field():
    s = FDTD_2D_Ez(1, 1, 20, 16, 1e9, Nt=1)
    s.mesh_from_density([1, 2, 1], [1, 3, 2])
    x, y = 0.413, 0.627
    indices, weights = point_stencil(s.mesh, x, y)
    X, Y = np.meshgrid(s.mesh.x, s.mesh.y, indexing="ij")
    field = 2 + 3 * X - 4 * Y + 5 * X * Y
    assert weights @ field.ravel()[indices] == pytest.approx(
        2 + 3 * x - 4 * y + 5 * x * y, abs=1e-13
    )


@cuda
@pytest.mark.cuda
def test_cpml_state_is_reset_and_residency_maintained():
    s = FDTD_2D_Ez(0.02, 0.02, 40, 40, 20e9, Nt=120)
    s.add_PML(6)
    s.add_source("point", x=0.01, y=0.01, width=1e-11, delay=2e-11)
    rx = s.add_receiver("line", x=0.012, y=(0.008, 0.012), samples=7)
    a, b = s.run(), s.run()
    np.testing.assert_array_equal(a.fields["Ez"], b.fields["Ez"])
    assert a.receivers[rx].shape == (120, 7)
    assert a.diagnostics["stepping_transfers"] == 0
    assert a.diagnostics["boundary"] == "CFS-CPML"


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("angle", [0, 30, 45, 90])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_cpml_reflection_against_enlarged_domain(angle, nonuniform):
    from benchmarks.validate_cpml import wavepacket_case

    report, *_ = wavepacket_case(angle, nonuniform, long_run=angle == 45, pec_control=angle == 45)
    # Predeclared acceptance: <1% waveform reflection error, <1% late field norm.
    assert report["peak_waveform_error"] < 0.01, report
    assert report["relative_waveform_l2"] < 0.01, report
    assert report["relative_interior_echo"] < 0.01, report
    if angle == 45:
        assert report["late_finite"] and report["late_relative_field"] < 0.01, report
        assert report["pec_control_peak_error"] > 0.1, report


def test_time_resampling_and_no_extrapolation():
    from fdtdmesh.simulation import SimulationResult

    times = np.array([1.0, 2.0, 3.0])
    result = SimulationResult(None, 1.0, times, {0: np.c_[times, 2 * times]}, {}, {}, {})
    np.testing.assert_allclose(result.resample(0, [1.25, 2.5]), [[1.25, 2.5], [2.5, 5.0]])
    with pytest.raises(ValueError, match="time range"):
        result.resample(0, [0, 1])
