from collections import Counter, defaultdict

from scattermesh import simple_candidate_tasks, simple_dielectric_pool


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
