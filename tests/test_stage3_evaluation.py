from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh.data.generate import make_scene
from fdtdmesh.data.schema import write_manifest
from fdtdmesh.evaluation import (
    EvaluationConfig,
    compare_observables,
    converge_reference,
    evaluate_dataset,
)
from fdtdmesh.evaluation.metrics import sample_observables
from fdtdmesh.evaluation.pipeline import run_scene
from fdtdmesh.solver.tmz import cuda_available


def test_metrics_known_amplitude_phase_and_quiet_floor():
    t = np.linspace(0, 1, 2049)
    ref = np.sin(2 * np.pi * 8 * t)[:, None]
    out = compare_observables(1.1 * ref, ref, t, [8, 9])
    assert out["waveform_l2"] == pytest.approx(0.1)
    assert out["spectrum_l2"] == pytest.approx(0.1)
    shifted = np.sin(2 * np.pi * 8 * t + np.pi / 6)[:, None]
    phase = compare_observables(shifted, ref, t, [8])
    assert phase["phase_rms_degrees"] == pytest.approx(30, abs=0.01)
    quiet = compare_observables(np.zeros_like(ref), np.zeros_like(ref), t, [8])
    assert quiet["waveform_l2"] == 0 and quiet["phase_rms_degrees"] is None
    with pytest.raises(ValueError, match="Nyquist"):
        compare_observables(ref, ref, t, [3000])
    with pytest.raises(ValueError, match="finite"):
        compare_observables(ref * np.nan, ref, t, [8])


def test_observations_align_dt_and_use_known_zero_initial_state():
    times = np.linspace(0, 1, 11)
    for n in (20, 40):
        raw = np.linspace(1 / n, 1, n)
        result = SimpleNamespace(times=raw, receivers={0: raw[:, None]})
        np.testing.assert_allclose(sample_observables(result, times)[:, 0], times)
    with pytest.raises(ValueError, match="extrapolation"):
        sample_observables(result, [0, 2])


def test_convergence_requires_consecutive_passes_and_rejects_exhaustion():
    scene = make_scene(1, "train", 1)
    config = EvaluationConfig(reference_levels=(8, 16, 32, 64), relative_tolerance=0.02)

    def runner(values):
        def run(spec, budget, config, **kw):
            t = np.linspace(0, spec.t_end, config.samples)[1:]
            scale = values.pop(0)
            return SimpleNamespace(
                times=t,
                receivers={0: scale * np.sin(2 * np.pi * t / spec.t_end)[:, None]},
                diagnostics={},
            )

        return run

    status, _ = converge_reference(scene, config, runner=runner([1, 1.001, 1.1, 1.1001]))
    assert status["status"] == "nonconverged" and status["accepted_budget"] is None
    status, _ = converge_reference(scene, config, runner=runner([1, 1.1, 1.1001, 1.1002]))
    assert status["status"] == "converged" and len(status["levels"]) == 4

    def failure(*args, **kwargs):
        raise RuntimeError("resource limit")

    status, _ = converge_reference(scene, config, runner=failure)
    assert status["status"] == "failed" and "resource limit" in status["levels"][0]["error"]


def test_resource_limit_prevents_cuda_call():
    scene = make_scene(2, "train", 1)
    with pytest.raises(RuntimeError, match="resource limit"):
        run_scene(scene, [64, 64], EvaluationConfig(max_cell_updates=1))


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_real_cuda_dataset_reference_and_three_baselines(tmp_path):
    scene = make_scene(42, "test_iid", 1)
    scene.budgets = [[64, 64]]
    manifest = tmp_path / "dataset.json"
    write_manifest(manifest, [scene], generation={"seed": 42})
    output = tmp_path / "evaluated"
    report = evaluate_dataset(manifest, output, demo_cnn=True)
    assert report["scenes"][0]["status"] == "converged"
    assert report["checkpoint"]["untrained"]
    assert len(report["candidates"]) == 3
    assert all(c["status"] == "ok" for c in report["candidates"])
    assert all(c["diagnostics"]["stepping_transfers"] == 0 for c in report["candidates"])
    with np.load(output / scene.scene_id / "reference_latest.npz", allow_pickle=False) as saved:
        assert saved["waveforms"].shape == (1025, 3)
    assert (output / "summary.md").exists()
    with pytest.raises(ValueError, match="empty"):
        evaluate_dataset(manifest, output)


def test_nonconverged_reference_never_scores_candidates(tmp_path, monkeypatch):
    from fdtdmesh.evaluation import pipeline

    spec = make_scene(8, "test_iid", 1)
    manifest = tmp_path / "input.json"
    write_manifest(manifest, [spec], generation={"seed": 8})
    monkeypatch.setattr(
        pipeline,
        "converge_reference",
        lambda *a, **k: ({"status": "nonconverged", "levels": [], "accepted_budget": None}, None),
    )

    def forbidden(*a, **k):
        pytest.fail("Rejected reference must not trigger a candidate run")

    monkeypatch.setattr(pipeline, "run_scene", forbidden)
    report = evaluate_dataset(manifest, tmp_path / "output")
    assert report["scenes"][0]["status"] == "nonconverged" and not report["candidates"]
