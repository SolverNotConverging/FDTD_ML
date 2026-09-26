import json

import numpy as np

from scattermesh import c3_qualification_v2, c5_qualification_v2


def test_c5_scenes_match_areas_and_cover_counts_and_arrangements():
    scenes = c5_qualification_v2.generate_c5_controlled_scenes()
    assert len(scenes) == 6
    assert {scene["object_count"] for scene in scenes} == {3, 5}
    assert {scene["layout"] for scene in scenes} == {"compact", "aligned", "dispersed"}
    for count in (3, 5):
        group = [scene for scene in scenes if scene["object_count"] == count]
        assert len(group) == 3
        assert all(scene["stage"] == "C5" for scene in group)
        assert all(
            abs(scene["scene_metrics"]["occupied_area_fraction"] * 1.2**2 - 0.0068) < 1e-12
            for scene in group
        )
        assert all(scene["scene_metrics"]["minimum_axis_aligned_bbox_gap_m"] > 0 for scene in group)
    three_area = scenes[0]["scene_metrics"]["occupied_area_fraction"]
    five_area = scenes[3]["scene_metrics"]["occupied_area_fraction"]
    np.testing.assert_allclose(three_area, five_area, rtol=1e-12, atol=0)


def test_extended_c5_scenes_match_area_and_separate_all_objects():
    scenes = c5_qualification_v2.generate_c5_controlled_scenes((6, 8, 10))
    assert len(scenes) == 9
    for count in (6, 8, 10):
        group = [scene for scene in scenes if scene["object_count"] == count]
        assert {scene["layout"] for scene in group} == {"compact", "aligned", "dispersed"}
        for scene in group:
            assert len(scene["objects"]) == count
            assert scene["minimum_bbox_gap_m"] > 0
            assert abs(scene["scene_metrics"]["occupied_area_fraction"] * 1.2**2 - 0.0068) < 1e-12


def test_c5_campaign_writes_physical_and_layout_report(tmp_path, monkeypatch):
    def fake_evaluate_case(scene, cells, incidence_angle, policy, output, **kwargs):
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
    monkeypatch.setattr(c5_qualification_v2, "evaluate_case", fake_evaluate_case)
    report = c5_qualification_v2.run_c5_layout_qualification(
        tmp_path,
        reference_levels=(64, 96),
        budgets=(32,),
        policies=("uniform", "center"),
    )
    assert report["qualified_scene_count"] == 6
    assert report["candidate_case_count"] == 12
    assert report["candidate_status_counts"] == {"accepted": 12}
    assert len(report["observed_area_by_object_count_m2"]["3"]) == 3
    assert len(report["observed_area_by_object_count_m2"]["5"]) == 3
    assert (tmp_path / "c5_layout_qualification_report.json").is_file()
