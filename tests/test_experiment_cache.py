"""Reuse requires an exact manifest match, never only a friendly folder name."""

from copy import deepcopy

import pytest

from fdtdmesh.benchmarks.common import experiment_directory, read_json, write_json


@pytest.mark.parametrize(
    "name,value",
    [
        ("cells", [80, 80]),
        ("seed", 9),
        ("max_evaluations", 120),
        ("max_seconds", 300),
        ("feature_anchors", True),
    ],
)
def test_changed_settings_select_an_independent_directory(tmp_path, name, value):
    description = dict(
        schema=1,
        optimizer=dict(
            cells=[192, 192], seed=0, max_evaluations=60, max_seconds=600, feature_anchors=False
        ),
    )
    first = experiment_directory(tmp_path, description)
    assert read_json(first / "experiment.json") == description
    assert experiment_directory(tmp_path, deepcopy(description)) == first
    changed = deepcopy(description)
    changed["optimizer"][name] = value
    second = experiment_directory(tmp_path, changed)
    assert first != second
    assert experiment_directory(tmp_path, changed) == second
    assert read_json(first / "experiment.json") == description


def test_legacy_and_mismatching_manifests_are_not_reused(tmp_path):
    legacy = tmp_path / "optimization.h5"
    legacy.write_bytes(b"legacy archive")
    description = dict(schema=1, solver=dict(precision="float64"))
    first = experiment_directory(tmp_path, description)
    assert first != tmp_path
    write_json(first / "experiment.json", dict(schema=1, solver=dict(precision="float32")))
    second = experiment_directory(tmp_path, description)
    assert first != second
    assert experiment_directory(tmp_path, description) == second
    assert legacy.read_bytes() == b"legacy archive"
