"""Physical and implementation checks; skips explicitly when CUDA is unavailable."""

import time
from dataclasses import replace

import numpy as np
import pytest

from fdtdmesh.cases import canonical_case
from fdtdmesh.constants import C0
from fdtdmesh.mesh import Mesh
from fdtdmesh.scattering import cylinder_amplitude, far_field, make_contour, pattern_error
from fdtdmesh.scene import Scene2D
from fdtdmesh.simulation import Convergence, ConvergenceError, prepare, run_scattering
from fdtdmesh.solver.coefficients import build_coefficients
from fdtdmesh.solver.reference_tmz import run_reference
from fdtdmesh.solver.tmz import cuda_backend

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def runtime():
    try:
        return cuda_backend()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("graded", [False, True])
def test_cuda_fields_and_staggered_dft_match_numpy(runtime, dtype, graded):
    case, mesh = canonical_case(ppw=16)
    if graded:

        def warp(lines, strength):
            z = lines / (C0 / case.frequency)
            return lines + (C0 / case.frequency) * np.where(
                (z > 1.5) & (z < 4.5), strength * np.sin(2 * np.pi * (z - 1.5) / 3), 0
            )

        mesh = Mesh(warp(mesh.x, 0.10), warp(mesh.y, 0.08))
    c, box, source, contour, options = prepare(case, mesh, dtype=dtype)
    nt = 173  # Deliberately not a multiple of the graph unroll or checkpoint interval.
    options.update(max_steps=nt, check_interval=64, auto_stop=False, debug=True)
    rng = np.random.default_rng(12)
    initial = (
        rng.normal(0, 0.01, (mesh.Nx + 1, mesh.Ny + 1)),
        rng.normal(0, 1e-5, (mesh.Nx + 1, mesh.Ny)),
        rng.normal(0, 1e-5, (mesh.Nx, mesh.Ny + 1)),
    )
    gpu = runtime.run(c, (mesh.Nx, mesh.Ny), box, source, contour, options, initial=initial)
    u = np.arange(1, nt + 1) * c.dt * case.frequency - options["pulse_delay"]
    wave = (
        options["source_scale"]
        * np.exp(-((u / options["pulse_width"]) ** 2))
        * np.cos(2 * np.pi * u)
    )
    cpu = run_reference(
        c, (mesh.Nx, mesh.Ny), box, wave, source, contour, case.frequency, initial=initial
    )
    tol = 1e-5 if dtype == "float32" else 2e-13
    for key, expected in zip(["Ez", "Hx", "Hy", "currents", "incident"], cpu[:5]):
        actual = gpu[4][key]
        if actual.ndim == expected.ndim + 1:
            actual = actual[0]
        assert np.linalg.norm(actual - expected) <= tol * np.linalg.norm(expected)
    assert gpu[3]["Nt"] == nt
    assert gpu[2][-1, 0] == nt


def test_gpu_nf2ff_multibin_and_residency(runtime):
    case, mesh = canonical_case(
        ppw=24, frequencies=(0.9e9, 1e9, 1.1e9), convergence=Convergence(check_interval=256)
    )
    result = run_scattering(case, mesh, diagnostic_download=True)
    _, _, _, contour, options = prepare(case, mesh)
    for f, freq in enumerate(result.frequencies):
        current = result.debug["currents"][f]
        amplitude, width = far_field(
            contour,
            current[:, 0],
            current[:, 1],
            freq,
            case.angles,
            result.debug["incident"][f, options["origin_index"]],
            case.origin,
        )
        np.testing.assert_allclose(result.amplitude[f], amplitude, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.width[f], width, rtol=1e-12, atol=1e-12)
        exact = cylinder_amplitude(0.47 * C0 / 1e9, freq, case.angles)
        assert pattern_error(result.width[f], 4 / (2 * np.pi * freq / C0) * abs(exact) ** 2) < 0.03
    assert result.diagnostics["result_d2h_bytes"] == 3 * len(case.angles) * 3 * 8
    assert result.diagnostics["stepping_field_transfers"] == 0


def test_device_stop_guard_stability_and_safety_cap(runtime):
    case, mesh = canonical_case(ppw=16, convergence=Convergence(check_interval=256))
    result = run_scattering(case, mesh)
    minimum = prepare(case, mesh)[-1]["min_steps"]
    assert result.converged and result.history[-1, 4] == 3
    assert result.history[-3:, 0].min() >= minimum
    assert np.all(result.history[-3:, 2] < 1)
    assert np.all(result.history[-3:, 3] < case.convergence.field_tol)
    assert result.debug == {} and result.diagnostics["debug_d2h_bytes"] == 0
    cap = minimum + 1
    failed = replace(case, convergence=Convergence(max_steps=cap, check_interval=256))
    with pytest.raises(ConvergenceError) as error:
        run_scattering(failed, mesh)
    assert error.value.result.diagnostics["status"] == "max_steps"
    assert error.value.result.diagnostics["Nt"] == cap


def test_empty_domain_and_cylinder_refinement(runtime):
    empty, mesh = canonical_case("empty", ppw=16, convergence=Convergence(check_interval=256))
    assert abs(run_scattering(empty, mesh).amplitude).max() < 1e-12
    errors = []
    for ppw in (16, 32):
        case, mesh = canonical_case(ppw=ppw, convergence=Convergence(check_interval=256))
        result = run_scattering(case, mesh)
        exact = cylinder_amplitude(0.47 * C0 / case.frequency, case.frequency, case.angles)
        errors.append(
            pattern_error(result.width[0], 4 / (2 * np.pi * case.frequency / C0) * abs(exact) ** 2)
        )
    assert errors[1] < errors[0] / 2 and errors[1] < 0.007


def test_slow_progress_callback_does_not_extend_gpu_execution(runtime):
    case, mesh = canonical_case(ppw=16, convergence=Convergence(check_interval=256))
    c, box, source, contour, options = prepare(case, mesh)
    options.update(max_steps=30000, auto_stop=False)
    baseline = runtime.run(c, (mesh.Nx, mesh.Ny), box, source, contour, options)
    reports = []

    def slow(report):
        reports.append(report)
        time.sleep(0.4)

    observed = runtime.run(c, (mesh.Nx, mesh.Ny), box, source, contour, options, progress=slow)
    assert reports[-1]["step"] == 30000
    assert all(a["generation"] < b["generation"] for a, b in zip(reports, reports[1:]))
    assert observed[3]["gpu_ms"] < baseline[3]["gpu_ms"] * 2 + 100
    np.testing.assert_array_equal(observed[0], baseline[0])


def test_malformed_native_peer_is_rejected(runtime):
    case, mesh = canonical_case(ppw=16)
    c, box, source, contour, options = prepare(case, mesh)
    c.hx.peer[0, 0] = c.hx.peer.size
    with pytest.raises(ValueError, match="partner"):
        runtime.run(c, (mesh.Nx, mesh.Ny), box, source, contour, options)


def test_tiny_cut_lossless_long_run_is_bounded(runtime):
    mesh = Mesh(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
    scene = Scene2D(1, 1).add_rectangle((0.375 + 1e-10, 0.643), (0.317, 0.693))
    c = build_coefficients(scene, mesh)
    assert min(c.hy.length[c.hy.length > 0]) < 2e-10
    assert c.dt > 0.5 * c.dt_cartesian
    rng = np.random.default_rng(4)
    e = rng.normal(size=(33, 33))
    e[c.pec.astype(bool)] = 0
    contour = make_contour(mesh, (0.125, 0.875, 0.125, 0.875))
    options = dict(
        max_steps=20003,
        check_interval=256,
        min_steps=0,
        stable_checks=3,
        auto_stop=False,
        debug=True,
        frequency=1e9,
        frequencies=(1e9,),
        angles=(0.0,),
        pulse_width=1.5,
        pulse_delay=6.0,
        source_scale=0.0,
        origin=(0.5, 0.5),
        origin_index=16,
        rtol=1e-5,
        atol=1e-8,
        field_tol=1e-5,
    )
    result = runtime.run(
        c,
        (32, 32),
        (8, 24, 8, 24),
        2,
        contour,
        options,
        initial=(e, np.zeros((33, 32)), np.zeros((32, 33))),
    )
    assert result[3]["Nt"] == 20003
    assert np.isfinite(result[4]["Ez"]).all()
    assert np.linalg.norm(result[4]["Ez"]) < 2 * np.linalg.norm(e)
