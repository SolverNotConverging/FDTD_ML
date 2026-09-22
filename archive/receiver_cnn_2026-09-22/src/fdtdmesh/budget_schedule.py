"""Reproducible per-scene budgets and globally reserved validation combinations."""

import numpy as np

from .mesh import cell_count

FIXED_BUDGETS = ((48, 48), (64, 64), (96, 96), (128, 128))
HELDOUT_BUDGETS = ((80, 80), (112, 112), (72, 104), (104, 72))


def normalize_budgets(values):
    """Accept square counts or explicit (Nx, Ny) pairs, without duplicates."""
    if values is None:
        return None
    result = []
    for value in values:
        if np.ndim(value) == 0:
            pair = (cell_count(value),) * 2
        else:
            if len(value) != 2:
                raise ValueError("Each budget must contain Nx and Ny")
            pair = tuple(cell_count(v) for v in value)
        if pair in result:
            raise ValueError("budgets must be distinct")
        result.append(pair)
    if not result:
        raise ValueError("budgets must not be empty")
    return tuple(result)


def mixed_budget_plan(scenes, seed=2026):
    assignments = {}
    for scene in sorted(scenes, key=lambda s: s.scene_id):
        if scene.split not in ("train", "validation"):
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, scene.seed, 1]))
        pairs = [(pair, "fixed_square") for pair in FIXED_BUDGETS]
        # One intermediate resolution in each half of the range.
        excluded = {48, 64, 80, 96, 112, 128}
        for lo, hi in ((49, 88), (88, 128)):
            n = int(rng.choice([v for v in range(lo, hi) if v not in excluded]))
            pairs.append(((n, n), "random_square"))
        while True:
            nx, ny = map(int, rng.integers(48, 129, size=2))
            if nx != ny and (nx, ny) not in HELDOUT_BUDGETS and (ny, nx) not in HELDOUT_BUDGETS:
                break
        pairs.extend([((nx, ny), "rectangular"), ((ny, nx), "rectangular")])
        if scene.split == "validation":
            pairs.extend((pair, "heldout_budget") for pair in HELDOUT_BUDGETS)
        assignments[scene.scene_id] = [
            dict(budget=list(pair), group=group) for pair, group in pairs
        ]
    return dict(
        version=1,
        seed=seed,
        minimum=48,
        maximum=128,
        heldout_budgets=[list(pair) for pair in HELDOUT_BUDGETS],
        assignments=assignments,
    )
