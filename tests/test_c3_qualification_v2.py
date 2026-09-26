import json

import numpy as np

from scattermesh import c3_qualification_v2
from scattermesh.curriculum_v2 import scene_metrics


def test_controlled_c3_scenes_vary_pair_controls_with_positive_gaps():
    scenes = c3_qualification_v2.generate_c3_pair_sweep()
    assert len(scenes) == 4
    assert {scene["stage"] for scene in scenes} == {"C3"}
    assert {scene["control_name"] for scene in scenes} == {
        "separation_near",
        "separation_far",
        "size_ratio_2",
        "diagonal_pair",
    }
    assert max(scene["pair_size_ratio"] for scene in scenes) == 2
    assert scenes[-1]["pair_orientation_rad"] != 0
    for scene in scenes:
        assert len(scene["objects"]) == 2
        assert scene_metrics(scene)["minimum_axis_aligned_bbox_gap_m"] > 0


def test_c3_campaign_writes_resumable_reference_and_mesh_report(tmp_path, monkeypatch):
    def fake_evaluate_case(
        scene,
        cells,
        incidence_angle,
        policy,
        output,
        *,
        reference=None,
        **kwargs,
    ):
        output.mkdir(parents=True, exist_ok=True)
        field = np.ones((3, 12), dtype=np.complex128)
        np.savez_compressed(output / "spectra.npz", complex_far_field=field)
        record = {
            "status": "accepted",
            "accepted": True,
            "dt": 1e-11,
            "Nt": 10,
            "wall_seconds": 0.01,
            "attempts": [{"duration_s": 10e-9}],
            "complex_relative_l2": 0.0,
            "width_relative_l2": 0.0,
            "joint_scattering_loss": 0.0,
            "uniform_repair_fraction": [0.0, 0.0],
        }
        (output / "record.json").write_text(json.dumps(record))
        return record, False

    monkeypatch.setattr(c3_qualification_v2, "evaluate_case", fake_evaluate_case)
    report = c3_qualification_v2.run_c3_pair_qualification(
        tmp_path,
        reference_levels=(64, 96),
        budgets=(32,),
        policies=("uniform", "interface"),
    )
    assert report["qualified_scene_count"] == 4
    assert report["candidate_case_count"] == 8
    assert report["candidate_status_counts"] == {"accepted": 8}
    assert (tmp_path / "c3_pair_qualification_report.json").is_file()
    assert all(scene["qualification"]["selected_cells"] == 96 for scene in report["scenes"])
