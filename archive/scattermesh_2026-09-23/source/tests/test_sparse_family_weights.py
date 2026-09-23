import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cluster_finetune_weights_preserve_pair_and_simple_controls():
    trainer = _script("train_mesh_distillation.py")
    launcher = _script("launch_nine_model_grid.py")
    weights = launcher.parse_family_weights([
        "simple=0.20", "sparse_pair=0.35", "sparse_cluster=0.45",
    ])
    examples = [
        {"family": "simple_remediation"},
        {"family": "sparse_pair"},
        {"family": "sparse_pair"},
        {"family": "sparse_cluster"},
        {"family": "sparse_cluster"},
    ]
    actual = trainer._validation_weights(SimpleNamespace(examples=examples), weights).numpy()
    assert actual[0] == pytest.approx(0.20)
    assert sum(actual[1:3]) == pytest.approx(0.35)
    assert sum(actual[3:]) == pytest.approx(0.45)
    assert all(config["family_sampling_weights"] == weights for config in launcher.grid_configs(weights))


def test_legacy_sparse_weighting_stays_unchanged():
    trainer = _script("train_mesh_distillation.py")
    launcher = _script("launch_nine_model_grid.py")
    examples = [
        {"family": "simple"}, {"family": "sparse_pair"}, {"family": "sparse_cluster"},
    ]
    actual = trainer._validation_weights(
        SimpleNamespace(examples=examples), {"simple": 0.30, "sparse": 0.70}
    ).numpy()
    assert actual == pytest.approx([0.30, 0.35, 0.35])
    assert next(launcher.grid_configs())["family_sampling_weights"] == {"simple": 0.30, "sparse": 0.70}


def test_cluster_weight_parser_rejects_missing_or_nonunit_fractions():
    parser = _script("launch_nine_model_grid.py").parse_family_weights
    with pytest.raises(ValueError):
        parser(["simple=0.2", "sparse_pair=0.8"])
    with pytest.raises(ValueError):
        parser(["simple=0.2", "sparse_pair=0.35", "sparse_cluster=0.40"])
