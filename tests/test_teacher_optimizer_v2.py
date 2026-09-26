import numpy as np
import pytest

from scattermesh.teacher_optimizer_v2 import density_axes, promote_optimized_candidates


def test_smooth_density_projection_has_exact_budget_and_grading():
    weights = np.r_[np.linspace(-1.5, 1.5, 6), np.linspace(1.5, -1.5, 6)]
    axes, repairs = density_axes(weights, 48)
    for axis in axes:
        widths = np.diff(axis)
        assert len(widths) == 48
        assert axis[0] == 0.0
        assert axis[-1] == 1.2
        assert np.all(widths > 0)
        assert np.max(np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])) <= 3.0 + 1e-9
    assert all(0 <= repair <= 1 for repair in repairs)


def test_density_search_rejects_out_of_bounds_or_nonfinite_weights():
    with pytest.raises(ValueError):
        density_axes(np.zeros(11), 48)
    with pytest.raises(ValueError):
        density_axes(np.r_[np.full(11, 0.0), 1.6], 48)
    with pytest.raises(ValueError):
        density_axes(np.full(12, np.nan), 48)


def test_only_completed_valid_searches_promote_training_targets(tmp_path):
    search = tmp_path / "search"
    source = search / "trials" / "trial_0000"
    source.mkdir(parents=True)
    (source / "record.json").write_text(
        '{"status":"accepted","accepted":true,"joint_scattering_loss":0.01,"dt":1e-11}'
    )
    np.savez(source / "spectra.npz", x=np.linspace(0, 1.2, 49), y=np.linspace(0, 1.2, 49))
    trial = {"index": 0, "accepted": True, "teacher_score": 0.01, "weights": [0] * 12}
    partial = promote_optimized_candidates(
        {"complete": False, "trials": [trial]}, search, tmp_path / "partial"
    )
    assert all(row["status"] == "incomplete" for row in partial)
    finished = promote_optimized_candidates(
        {"complete": True, "trials": [trial]}, search, tmp_path / "finished"
    )
    assert finished[0]["status"] == "accepted"
    assert finished[1]["status"] == "no_valid_optimized_candidate"
    with np.load(tmp_path / "finished" / "optimized_0" / "spectra.npz") as arrays:
        assert len(arrays["x"]) == 49
