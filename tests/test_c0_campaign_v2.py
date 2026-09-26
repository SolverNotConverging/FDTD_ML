"""Guard the C0 lineage split and frozen label conditions."""

from collections import Counter

from scattermesh.c0_campaign_v2 import c0_scenes, make_manifest
from scattermesh.campaign_v2 import _conditions_for_scene


def test_c0_acquisition_materials_and_shapes_in_every_split(tmp_path):
    scenes = c0_scenes(128)
    assert len({scene["lineage_id"] for scene in scenes}) == 128
    for split, expected in (("train", 96), ("validation", 16), ("test", 16)):
        group = [scene for scene in scenes if scene["split"] == split]
        assert len(group) == expected
        assert Counter(scene["material"]["kind"] for scene in group) == {
            "dielectric": expected * 3 // 4,
            "pec": expected // 4,
        }
        assert {scene["family"] for scene in group} == {"circle", "ellipse", "rectangle"}
        assert {
            scene["material"]["epsilon_r"]
            for scene in group if scene["material"]["kind"] == "dielectric"
        } == {2, 4, 8, 12}

    manifest = make_manifest(tmp_path / "manifest.json", count=128)
    assert sum(map(len, manifest["conditions_by_scene"].values())) == 448
    assert all(
        scene["lineage_id"] not in manifest["conditions_by_scene"]
        for scene in manifest["scenes"] if scene["split"] == "test"
    )
    assert all(
        _conditions_for_scene(manifest, scene) == []
        for scene in manifest["scenes"] if scene["split"] == "test"
    )
