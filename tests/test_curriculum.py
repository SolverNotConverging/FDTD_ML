from collections import Counter, defaultdict

from scattermesh import (
    circle_remediation_pool,
    factorial_simple_dielectric_pool,
    simple_candidate_tasks,
    simple_dielectric_pool,
)
from scattermesh.curriculum import REMEDIATION_ANGLES


def test_simple_pool_has_grouped_splits_and_resolved_features():
    pool = simple_dielectric_pool()
    assert len(pool) == 32
    assert len({row["geometry_id"] for row in pool}) == 32
    assert Counter(row["split"] for row in pool) == {
        "train": 24,
        "validation": 4,
        "test": 4,
    }
    lineage_splits = defaultdict(set)
    for row in pool:
        lineage_splits[row["lineage_id"]].add(row["split"])
        assert row["feature_cells_on_256_input"] >= 18
        assert row["minimum_internal_wavelength_m"] > 0
    assert len(lineage_splits) == 8
    assert all(len(splits) == 1 for splits in lineage_splits.values())


def test_candidate_tasks_preserve_geometry_lineage_and_hold_out_angles():
    pool = simple_dielectric_pool()
    tasks = simple_candidate_tasks(pool, budgets=(32, 48, 64))
    assert len(tasks) == 264
    assert len({task["task_id"] for task in tasks}) == len(tasks)
    by_geometry = defaultdict(list)
    for task in tasks:
        by_geometry[task["geometry_id"]].append(task)
    geometry = {row["geometry_id"]: row for row in pool}
    for geometry_id, rows in by_geometry.items():
        expected = geometry[geometry_id]
        assert {row["split"] for row in rows} == {expected["split"]}
        assert {row["lineage_id"] for row in rows} == {expected["lineage_id"]}
    train_angles = {task["incidence_angle_rad"] for task in tasks if task["split"] == "train"}
    held_out_angles = {task["incidence_angle_rad"] for task in tasks if task["split"] != "train"}
    assert train_angles.isdisjoint(held_out_angles)


def test_factorial_pool_decorrelates_material_size_and_loss():
    pool = factorial_simple_dielectric_pool()
    assert len(pool) == 148
    assert Counter(row["split"] for row in pool) == {
        "train": 92,
        "validation": 24,
        "test": 32,
    }
    lineages = defaultdict(list)
    for row in pool:
        lineages[row["lineage_id"]].append(row)
        assert row["feature_cells_on_256_input"] >= 20
    assert Counter(rows[0]["split"] for rows in lineages.values()) == {
        "train": 46,
        "validation": 12,
        "test": 16,
    }
    assert all(len(rows) == 2 for rows in lineages.values())

    train_main_factors = {
        (
            row["epsilon_r"],
            round(row["radius_m"] / (0.97 if row["variant_index"] == 0 else 1.03), 6),
            row["sigma_e_s_per_m"],
        )
        for row in pool
        if row["split"] == "train" and row["sigma_e_s_per_m"] > 0
    }
    assert len(train_main_factors) == 4 * 3 * 2 + 3 * 3 * 2
    lossless_controls = {
        (row["epsilon_r"], round(row["radius_m"], 6))
        for row in pool
        if row["split"] == "train" and row["sigma_e_s_per_m"] == 0
    }
    assert {epsilon_r for epsilon_r, _ in lossless_controls} == {2.0, 4.0}
    assert max(row["epsilon_r"] for row in pool) == 30.0
    high_epsilon_train = [
        row for row in pool if row["split"] == "train" and row["epsilon_r"] > 10
    ]
    assert {round(row["loss_tangent_at_1ghz"], 2) for row in high_epsilon_train} == {
        0.10,
        0.20,
    }
    high_epsilon_by_split = {
        split: {
            row["epsilon_r"]
            for row in pool
            if row["split"] == split and row["epsilon_r"] > 10
        }
        for split in ("train", "validation", "test")
    }
    assert high_epsilon_by_split == {
        "train": {12.0, 20.0, 30.0},
        "validation": {16.0, 26.0},
        "test": {14.0, 28.0},
    }
    assert all(
        high_epsilon_by_split[left].isdisjoint(high_epsilon_by_split[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )


def test_factorial_pool_expands_to_grouped_conditions():
    pool = factorial_simple_dielectric_pool()
    tasks = simple_candidate_tasks(pool)
    assert len(tasks) == 1552
    assert Counter(task["split"] for task in tasks) == {
        "train": 1104,
        "validation": 192,
        "test": 256,
    }
    lineage_splits = defaultdict(set)
    geometry = {row["geometry_id"]: row for row in pool}
    for task in tasks:
        lineage_splits[task["lineage_id"]].add(task["split"])
        assert task["lineage_id"] == geometry[task["geometry_id"]]["lineage_id"]
    assert all(len(splits) == 1 for splits in lineage_splits.values())


def test_circle_remediation_pool_is_grouped_and_disjoint_from_frozen_gate():
    pool = circle_remediation_pool()
    assert len(pool) == 40
    assert Counter(row["split"] for row in pool) == {
        "train": 24,
        "validation": 8,
        "test": 8,
    }
    lineages = defaultdict(list)
    for row in pool:
        lineages[row["lineage_id"]].append(row)
        assert row["radius_m"] not in {0.040, 0.075, 0.135}
        assert tuple(row["center_m"]) not in {
            (0.60, 0.60),
            (0.38, 0.38),
            (0.82, 0.38),
            (0.38, 0.82),
            (0.82, 0.82),
        }
        assert set(row["incidence_angles_rad"]).isdisjoint({0.35, 4.9})
    assert len(lineages) == 20
    assert all(len(rows) == 2 for rows in lineages.values())
    assert all(len({row["split"] for row in rows}) == 1 for rows in lineages.values())
    assert all(
        set(REMEDIATION_ANGLES[left]).isdisjoint(REMEDIATION_ANGLES[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )

    tasks = simple_candidate_tasks(pool, budgets=(32, 48))
    assert len(tasks) == 256
    assert Counter(row["split"] for row in tasks) == {
        "train": 192,
        "validation": 32,
        "test": 32,
    }
