import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _builder():
    path = ROOT / "scripts/build_sparse_joint_dataset.py"
    spec = importlib.util.spec_from_file_location("sparse_joint_dataset_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copied_sparse_examples_recover_frozen_scene_families():
    enrich = _builder()._enrich_sparse_metadata
    campaign = json.loads((ROOT / "configs/sparse_pair_campaign_96.json").read_text())
    for scene in campaign["scenes"]:
        updated = enrich({
            "family": "sparse_pair", "geometry_id": scene["scene_id"],
            "objects": scene["objects"],
        })
        assert updated["scene_family_id"] == scene["family_id"]
        assert updated["pec_circle_count"] == sum(
            obj["shape"] == "circle" and obj["material"]["kind"] == "pec"
            for obj in scene["objects"]
        )


def test_legacy_pair_receives_explicit_control_family():
    enrich = _builder()._enrich_sparse_metadata
    old = {
        "family": "sparse_pair",
        "geometry_id": "close_equal_southwest",
        "objects": [{"shape": "circle", "material": {"kind": "pec"}}],
    }
    updated = enrich(old)
    assert updated["scene_family_id"] == "legacy_pair"
    assert updated["pec_circle_count"] == 1
    assert "scene_family_id" not in old
