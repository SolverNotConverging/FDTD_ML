"""Backend equivalence checks for the optional PyTorch dielectric solver."""

import numpy as np
import pytest

from scattermesh import (
    PEC,
    Circle,
    Grid,
    Material,
    PlaneWave,
    Rectangle,
    focused_axis,
    simulate,
    simulate_cuda,
)

torch = pytest.importorskip("torch")


def _problem():
    x = focused_axis(1.2, 32)
    grid = Grid(x, x)
    objects = [Circle((0.6, 0.6), 0.06, Material(4, 0.01))]
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    settings = dict(
        frequencies=[0.8e9, 1e9, 1.2e9],
        duration=20e-9,
        pml_thickness=0.15,
        samples=8,
    )
    return grid, objects, source, settings


def _assert_equivalent(reference, candidate, *, rtol, atol):
    for name in reference.fields:
        np.testing.assert_allclose(
            candidate.fields[name], reference.fields[name], rtol=rtol, atol=atol
        )
    np.testing.assert_allclose(
        candidate.monitor.electric, reference.monitor.electric, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        candidate.monitor.tangential_h,
        reference.monitor.tangential_h,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        candidate.monitor.incident, reference.monitor.incident, rtol=rtol, atol=atol
    )
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    np.testing.assert_allclose(
        candidate.monitor.normalized_far_field(angles),
        reference.monitor.normalized_far_field(angles),
        rtol=rtol,
        atol=atol,
    )


def test_torch_cpu_float64_matches_numpy_update_and_streaming_dft():
    grid, objects, source, settings = _problem()
    reference = simulate(grid, objects, source, **settings)
    candidate = simulate_cuda(grid, objects, source, device="cpu", dtype="float64", **settings)
    _assert_equivalent(reference, candidate, rtol=2e-12, atol=2e-14)
    assert candidate.diagnostics["dft_phase_method"] == "recurrence_with_analytic_reanchor"
    assert candidate.diagnostics["host_full_field_transfers"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not visible")
def test_cuda_float64_matches_numpy_reference():
    grid, objects, source, settings = _problem()
    reference = simulate(grid, objects, source, **settings)
    candidate = simulate_cuda(grid, objects, source, device="cuda:0", dtype="float64", **settings)
    _assert_equivalent(reference, candidate, rtol=3e-12, atol=3e-14)
    assert candidate.diagnostics["backend"] == "torch_cuda"


@pytest.mark.parametrize("pec_mode", ["staircase", "conformal", "enlarged"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA device not visible"
            ),
        ),
    ],
)
def test_torch_pec_modes_match_numpy(pec_mode, device):
    axis = np.linspace(0, 1.2, 33)
    grid = Grid(axis, axis)
    objects = [Rectangle((0.388, 0.812, 0.401, 0.799), PEC())]
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    settings = dict(
        frequencies=[1e9],
        duration=12e-9,
        pml_thickness=0.15,
        pec_mode=pec_mode,
    )
    reference = simulate(grid, objects, source, **settings)
    candidate = simulate_cuda(grid, objects, source, device=device, dtype="float64", **settings)
    _assert_equivalent(reference, candidate, rtol=3e-12, atol=3e-14)
    assert candidate.diagnostics["pec_mode"] == pec_mode
    assert candidate.diagnostics["pec_boundary_edge_count"] > 0
    assert candidate.diagnostics["dt_fraction_of_grid_cfl"] == pytest.approx(
        reference.diagnostics["dt_fraction_of_grid_cfl"]
    )
    if pec_mode == "enlarged":
        assert candidate.diagnostics["pec_enlarged_nodes"] > 0


@pytest.mark.parametrize("pec_mode", ["conformal", "enlarged"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA device not visible"
            ),
        ),
    ],
)
def test_torch_separated_mixed_materials_match_numpy(pec_mode, device):
    axis = np.linspace(0, 1.2, 33)
    grid = Grid(axis, axis)
    objects = [
        Circle((0.47, 0.59), 0.07, PEC()),
        Circle((0.73, 0.62), 0.06, Material(4, 0.02)),
    ]
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    settings = dict(
        frequencies=[0.8e9, 1e9, 1.2e9],
        duration=12e-9,
        pml_thickness=0.15,
        pec_mode=pec_mode,
    )
    reference = simulate(grid, objects, source, **settings)
    candidate = simulate_cuda(grid, objects, source, device=device, dtype="float64", **settings)
    _assert_equivalent(reference, candidate, rtol=3e-12, atol=3e-14)
    total = candidate.fields["Ez"] + source.electric(
        grid.x[:, None], grid.y[None, :], candidate.diagnostics["simulated_time"]
    )
    np.testing.assert_allclose(
        total[objects[0].contains(grid.x[:, None], grid.y[None, :])], 0, atol=1e-18
    )


def test_cuda_backend_rejects_overlapping_materials_and_invalid_controls():
    grid, _, source, settings = _problem()
    with pytest.raises(ValueError, match="Overlapping or touching"):
        simulate_cuda(
            grid,
            [
                Circle((0.56, 0.6), 0.08, PEC()),
                Circle((0.64, 0.6), 0.04, Material(4)),
            ],
            source,
            device="cpu",
            **settings,
        )
    with pytest.raises(ValueError, match="dtype"):
        simulate_cuda(grid, [], source, device="cpu", dtype="float16", **settings)
    with pytest.raises(ValueError, match="interval"):
        simulate_cuda(grid, [], source, device="cpu", phase_reanchor_interval=0, **settings)
