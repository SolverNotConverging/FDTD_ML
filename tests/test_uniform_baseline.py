from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh.data.generate_v1 import make_scene
from fdtdmesh.evaluation.pipeline import EvaluationConfig, run_scene
from fdtdmesh.physics import score_candidates
from fdtdmesh.sampling import point_stencil
from fdtdmesh.simulation import FDTD_2D_Ez
from fdtdmesh.solver.coefficients import build_coefficients
from fdtdmesh.solver.tmz import cuda_available
from fdtdmesh.uniform import uniform_scene


def example():
    original = make_scene(2, "train", 0)
    lx, ly = original.domain
    return replace(
        original,
        anchor_probes=True,
        pec_policy="rectangles_and_wires",
        geometry=[
            dict(
                kind="rectangle",
                material="PEC",
                x_position=[0.371 * lx, 0.63 * lx],
                y_position=[0.413 * ly, 0.68 * ly],
            ),
            dict(kind="pec_line", x=0.723 * lx, y=[0.27 * ly, 0.63 * ly]),
        ],
    )


def test_uniform_spacing_and_nearest_pec_mapping_preserve_scene():
    spec = example()
    before = spec.to_dict()
    scene = uniform_scene(spec, [32, 32])
    np.testing.assert_allclose(np.diff(scene.mesh.x), spec.domain[0] / 32)
    np.testing.assert_allclose(np.diff(scene.mesh.y), spec.domain[1] / 32)
    assert not scene.x_anchors and not scene.y_anchors
    assert spec.to_dict() == before
    for entry in scene.mesh.metadata["pec_snapping"]:
        if entry["kind"] == "polygon":
            for p, q in zip(entry["original"], entry["snapped"]):
                assert abs(p[0] - q[0]) <= spec.domain[0] / 64
                assert abs(p[1] - q[1]) <= spec.domain[1] / 64
                assert q[0] in scene.mesh.x and q[1] in scene.mesh.y
        else:
            wire = entry["snapped"]
            assert wire["x"] in scene.mesh.x
            assert all(y in scene.mesh.y for y in wire["y"])
            _, _, _, mask = scene.sample([wire["x"]], [wire["y"][0], wire["y"][1]])
            assert mask.all()
    for probe, _ in scene.sources:
        sites, weights = point_stencil(scene.mesh, probe.x, probe.y)
        assert np.isclose(weights.sum(), 1)
        coordinates = np.array([(scene.mesh.x[i // 33], scene.mesh.y[i % 33]) for i in sites])
        np.testing.assert_allclose(weights @ coordinates, [probe.x, probe.y])


@pytest.mark.parametrize(
    "budget, expected_cells",
    [
        ([51, 51], (6, 6)),
        ([76, 76], (9, 9)),  # 9.5 requested cells: midpoint tie goes thinner.
        ([52, 53], (6, 7)),
        ([114, 82], (14, 10)),
        ([127, 127], (16, 16)),
    ],
)
def test_uniform_pml_snapping_preserves_uniform_grid_and_cpml(budget, expected_cells):
    spec = example()
    before = spec.to_dict()
    scene = uniform_scene(spec, budget)
    dx, dy = spec.domain[0] / budget[0], spec.domain[1] / budget[1]

    np.testing.assert_allclose(np.diff(scene.mesh.x), dx)
    np.testing.assert_allclose(np.diff(scene.mesh.y), dy)
    assert not scene.x_anchors and not scene.y_anchors
    assert spec.to_dict() == before

    snapping = scene.mesh.metadata["pml_snapping"]
    assert (snapping["x"]["cells"], snapping["y"]["cells"]) == expected_cells
    assert scene.pml.x.cells == expected_cells[0]
    assert scene.pml.y.cells == expected_cells[1]
    assert abs(snapping["x"]["interface_shift"]) <= dx / 2 + 1e-12 * spec.domain[0]
    assert abs(snapping["y"]["interface_shift"]) <= dy / 2 + 1e-12 * spec.domain[1]

    coefficients = build_coefficients(scene, scene.mesh, dtype=scene.dtype)
    assert coefficients.cpml.ndim == 2 and coefficients.cpml.shape[1] == 3
    assert np.isfinite(coefficients.cpml).all()
    assert np.all((coefficients.cpml[:, 0] > 0) & (coefficients.cpml[:, 0] <= 1))
    assert np.all((coefficients.cpml[:, 1] > 0) & (coefficients.cpml[:, 1] <= 1))


def test_uniform_reference_rejects_misaligned_pml_and_aligned_budget_is_exact():
    spec = example()
    with pytest.raises(ValueError, match="Uniform reference budget must align PML interfaces"):
        spec.build([76, 76], reference=True)

    scene = uniform_scene(spec, [64, 64])
    snapping = scene.mesh.metadata["pml_snapping"]
    assert scene.pml.x.cells == scene.pml.y.cells == 8
    assert snapping["x"]["actual_thickness"] == spec.pml["thickness"][0]
    assert snapping["y"]["actual_thickness"] == spec.pml["thickness"][1]
    assert snapping["x"]["interface_shift"] == snapping["y"]["interface_shift"] == 0


def test_reference_remains_exact_and_collapsed_pec_is_reported():
    spec = example()
    with pytest.raises(ValueError, match="Uniform mesh cannot retain these anchors"):
        spec.build([32, 32], reference=True).mesh_uniform()
    tiny = dict(spec.geometry[0], x_position=[0.371 * spec.domain[0], 0.372 * spec.domain[0]])
    with pytest.raises(ValueError, match="collapses"):
        uniform_scene(replace(spec, geometry=[tiny]), [32, 32])


def test_uniform_and_quasi_uniform_dispatch_share_sampled_averaging(monkeypatch):
    captured = []

    def run(scene):
        coefficients = build_coefficients(scene, scene.mesh, dtype=scene.dtype)
        captured.append((scene, coefficients))
        return SimpleNamespace(fields={"Ez": np.zeros((1, 1))}, diagnostics={})

    monkeypatch.setattr(FDTD_2D_Ez, "run", run)
    spec = example()
    config = EvaluationConfig(material_averaging="sampled", max_cell_updates=10**12)
    for strategy in ("uniform", "quasi_uniform"):
        run_scene(spec, [64, 64], config, strategy=strategy)
    uniform, quasi = captured
    assert uniform[1].averaging["mode"] == quasi[1].averaging["mode"] == "sampled"
    assert not uniform[0].x_anchors and quasi[0].x_anchors
    assert all(x in quasi[0].mesh.x for x in quasi[0].x_anchors)
    assert not np.allclose(np.diff(quasi[0].mesh.x), spec.domain[0] / 64)

    # The uniform candidate exposes the physical PML adjustment alongside PEC
    # snapping so benchmark records explain any rounded absorber thickness.
    result = run_scene(spec, [76, 76], config, strategy="uniform")
    pml_snapping = result.diagnostics["pml_snapping"]
    assert pml_snapping["x"]["cells"] == pml_snapping["y"]["cells"] == 9
    assert pml_snapping["x"]["actual_thickness"] < spec.pml["thickness"][0]


def test_snapped_benchmark_cannot_be_selected_as_anchored_target():
    rows = [
        dict(
            name=name,
            status="ok",
            target_eligible=eligible,
            diagnostics=dict(cell_updates=100),
            metrics=dict(waveform_l2_max=error, spectrum_l2_max=error),
        )
        for name, error, eligible in (("uniform", 0.001, False), ("quasi_uniform", 0.1, True))
    ]
    chosen, front = score_candidates(rows, beta=0.02)
    assert chosen["name"] == "quasi_uniform" and front == ["quasi_uniform"]
    assert chosen["cost_ratio"] == 1


def test_horizontal_wire_and_midpoint_tie():
    spec = example()
    lx, ly = spec.domain
    wire = dict(kind="pec_line", y=16.5 * ly / 32, x=[8.5 * lx / 32, 20.5 * lx / 32])
    scene = uniform_scene(replace(spec, geometry=[wire]), [32, 32])
    mapped = scene.primitives[0][2]
    assert mapped.y == scene.mesh.y[16]
    assert mapped.x == (scene.mesh.x[8], scene.mesh.x[20])


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_snapped_pec_real_cuda_waveforms():
    spec = example()
    lx, ly = spec.domain
    spec = replace(
        spec,
        sources=[dict(spec.sources[0], x=0.23 * lx, y=0.27 * ly)],
        receivers=[dict(kind="point", x=0.24 * lx, y=0.74 * ly)],
    )
    config = EvaluationConfig(material_averaging="sampled", max_cell_updates=10**12)
    result = run_scene(spec, [64, 64], config, strategy="uniform")
    assert result.diagnostics["material_averaging"]["mode"] == "sampled"
    assert len(result.diagnostics["pec_snapping"]) == 2
    assert all(np.isfinite(values).all() for values in result.receivers.values())
    assert any(np.any(values != 0) for values in result.receivers.values())
    np.testing.assert_allclose(np.diff(result.mesh.x), lx / 64)
