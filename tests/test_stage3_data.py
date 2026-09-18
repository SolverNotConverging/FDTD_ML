import json
from copy import deepcopy

import numpy as np
import pytest

from fdtdmesh.data.generate import generate_dataset, make_scene
from fdtdmesh.data.schema import SPLITS, SceneSpec, read_manifest, validate_splits, write_manifest


def test_generation_reproducible_diverse_and_disjoint(tmp_path):
    a, b = generate_dataset(4, 7), generate_dataset(4, 7)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    assert set(s.split for s in a) == set(SPLITS)
    assert set(s.family for s in a) >= {
        "rectangle",
        "circle",
        "thin_line",
        "gap",
        "touching",
        "mixed",
        "triangle",
        "polygon",
    }
    validate_splits(a)
    for s in a:
        if s.split == "test_material_ood":
            assert all(g.get("material") == "dielectric" for g in s.geometry)
            assert s.materials[0]["epsilon_r"] >= 8
        if s.split == "test_budget_ood":
            assert not any(budget in [[64, 64], [96, 96]] for budget in s.budgets)
    path = tmp_path / "manifest.json"
    written = write_manifest(path, a, generation={"seed": 7})
    metadata, read = read_manifest(path)
    assert written["dataset_id"] == metadata["dataset_id"]
    assert read[0].to_dict() == a[0].to_dict()
    altered = json.loads(path.read_text())
    altered["scenes"][0]["f_max"] *= 2
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="hash"):
        read_manifest(path)


def test_split_lineage_and_near_duplicate_detection():
    a = make_scene(11, "train", 1)
    b = deepcopy(a)
    b.scene_id = "heldout"
    b.split = "test_iid"
    with pytest.raises(ValueError, match="lineage"):
        validate_splits([a, b])
    b.group_id = "other"
    b.geometry[0]["center"][0] += 1e-7 * a.domain[0]
    with pytest.raises(ValueError, match="Near-duplicate"):
        validate_splits([a, b])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version=99),
        lambda d: d.update(scene_id="../escape"),
        lambda d: d["sources"][0].update(normalization="field_increment"),
        lambda d: d["geometry"][0].update(radius=1e-8),
        lambda d: d.update(f_max=float("nan")),
    ],
)
def test_schema_rejects_invalid_contracts(mutation):
    value = make_scene(4, "train", 1).to_dict()
    mutation(value)
    with pytest.raises(ValueError):
        SceneSpec.from_dict(value)


def test_uniform_references_refine_collar_cells_and_retain_physical_scene():
    spec = make_scene(31, "train", 2)
    coarse, fine = spec.build([64, 64], reference=True), spec.build([128, 128], reference=True)
    coarse.mesh_uniform()
    fine.mesh_uniform()
    assert coarse.pml.x.thickness == fine.pml.x.thickness
    assert coarse.pml.x.cells == 8 and fine.pml.x.cells == 16
    np.testing.assert_allclose(coarse.mesh.x, fine.mesh.x[::2], rtol=1e-14)
    assert all(anchor in fine.mesh.x for anchor in coarse.x_anchors)
    candidate = spec.build([96, 96])
    assert candidate.pml.x.cells == 8
    with pytest.raises(ValueError, match="align PML"):
        spec.build([65, 65], reference=True)


def test_subpixel_gap_is_rejected_but_touching_is_allowed():
    scene = make_scene(19, "train", 3)
    a, b = scene.geometry
    b["x_position"][0] = a["x_position"][1] + scene.domain[0] / 1000
    with pytest.raises(ValueError, match="gap"):
        scene.validate()
    b["x_position"][0] = a["x_position"][1]
    scene.validate()
