from collections import Counter, defaultdict

from scattermesh import (
    factorial_simple_dielectric_pool,
    simple_candidate_tasks,
    simple_dielectric_pool,
)


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
    assert len(pool) == 80
    assert Counter(row["split"] for row in pool) == {
        "train": 48,
        "validation": 16,
        "test": 16,
    }
    lineages = defaultdict(list)
    for row in pool:
        lineages[row["lineage_id"]].append(row)
        assert row["feature_cells_on_256_input"] >= 20
    assert Counter(rows[0]["split"] for rows in lineages.values()) == {
        "train": 24,
        "validation": 8,
        "test": 8,
    }
    assert all(len(rows) == 2 for rows in lineages.values())

    train_factors = {
        (
            row["epsilon_r"],
            round(row["radius_m"] / (0.97 if row["variant_index"] == 0 else 1.03), 6),
            row["sigma_e_s_per_m"],
        )
        for row in pool
        if row["split"] == "train"
    }
    assert len(train_factors) == 4 * 3 * 2


def test_factorial_pool_expands_to_grouped_conditions():
    pool = factorial_simple_dielectric_pool()
    tasks = simple_candidate_tasks(pool)
    assert len(tasks) == 832
    assert Counter(task["split"] for task in tasks) == {
        "train": 576,
        "validation": 128,
        "test": 128,
    }
    lineage_splits = defaultdict(set)
    geometry = {row["geometry_id"]: row for row in pool}
    for task in tasks:
        lineage_splits[task["lineage_id"]].add(task["split"])
        assert task["lineage_id"] == geometry[task["geometry_id"]]["lineage_id"]
    assert all(len(splits) == 1 for splits in lineage_splits.values())
