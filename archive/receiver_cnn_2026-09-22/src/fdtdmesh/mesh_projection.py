"""Joint anchor assignment / line position optimization (normalized L1 objective)."""

from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

from .mesh import (
    AxisCollar,
    AxisConstraints,
    MeshInfeasibleError,
    MeshOptimizationError,
    cell_count,
    validate_spacing,
)


def density_quantiles(length, count, density, *, lower=0.0, upper=None):
    upper = length if upper is None else upper
    rho = np.asarray(density, dtype=float)
    if rho.ndim != 1 or not rho.size or not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError("Density must be a finite, positive 1-D array")
    edges = np.linspace(0, length, len(rho) + 1)
    # Integrate only the learned interval: arbitrary density in fixed collars
    # must not affect either the target or its floating-point dynamic range.
    widths = np.maximum(0, np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower))
    active = widths > 0
    rho = rho[active] / rho[active].max()
    cropped_edges = np.r_[lower, np.minimum(edges[1:][active], upper)]
    cdf = np.r_[0, np.cumsum(rho * widths[active])]
    if cdf[-1] <= 0 or np.any(np.diff(cdf) <= 0):
        raise MeshInfeasibleError("Density dynamic range cannot be resolved in float64")
    out = np.interp(np.linspace(0, cdf[-1], count + 1), cdf, cropped_edges)
    out[[0, -1]] = [lower, upper]
    return out


def project_axis(
    length,
    count,
    density,
    anchors=(),
    constraints=None,
    *,
    collar=None,
    return_diagnostics=False,
    time_limit=30.0,
):
    started = perf_counter()
    count = cell_count(count)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Axis length must be finite and positive")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    c, collar = constraints or AxisConstraints(), collar or AxisCollar()
    fixed = collar.fixed_lines(length, count)
    anchors = np.unique(np.r_[0.0, anchors, length])
    if not np.isfinite(anchors).all() or anchors[0] < 0 or anchors[-1] > length:
        raise ValueError("Anchors must be finite and inside the domain")
    if len(anchors) > count + 1:
        raise MeshInfeasibleError("There must be at least one cell between each pair of anchors")
    left, right = collar.thickness, length - collar.thickness
    free = []
    for anchor in anchors:
        if anchor in fixed.values():
            continue
        if not left < anchor < right:
            raise MeshInfeasibleError("Anchor conflicts with fixed PML collar lines")
        free.append(anchor / length)
    learned_count = count - 2 * collar.cells
    if len(free) >= learned_count:
        raise MeshInfeasibleError("Interior budget cannot represent all anchors")
    target = np.empty(count + 1)
    target[collar.cells : count - collar.cells + 1] = density_quantiles(
        length, learned_count, density, lower=left, upper=right
    )
    for index, value in fixed.items():
        target[index] = value
    normalized = target / length
    try:
        validate_spacing(target, c)
        unchanged = all(a in target for a in anchors)
    except MeshInfeasibleError:
        unchanged = False
    gap = 0.0
    if unchanged:
        out, status = target.copy(), "unchanged"
    else:
        # x coordinates, absolute deviations, then binary anchor-to-line assignments.
        npos, ndev = count + 1, count - 1
        choices, next_var = [], npos + ndev
        for k, _ in enumerate(free):
            candidates = list(
                range(collar.cells + 1 + k, count - collar.cells - (len(free) - 1 - k))
            )
            choices.append([(i, next_var + j) for j, i in enumerate(candidates)])
            next_var += len(candidates)
        nv = next_var
        cost = np.zeros(nv)
        cost[npos : npos + ndev] = 1 / max(1, learned_count - 1)
        lb, ub = np.zeros(nv), np.ones(nv)
        integrality = np.zeros(nv)
        integrality[npos + ndev :] = 1
        for index, value in fixed.items():
            lb[index] = ub[index] = value / length
        rows, cols, vals, lower, upper = [], [], [], [], []

        def constraint(terms, low=-np.inf, high=np.inf):
            row = len(lower)
            for col, value in terms.items():
                rows.append(row)
                cols.append(col)
                vals.append(value)
            lower.append(low)
            upper.append(high)

        min_h = max(c.min_spacing / length, 1e-9)
        max_h = 1 if c.max_spacing is None else c.max_spacing / length
        for i in range(count):
            constraint({i: -1, i + 1: 1}, min_h, max_h)
        for i in range(count - 1):
            r = c.max_ratio
            constraint({i: r, i + 1: -1 - r, i + 2: 1}, high=0)
            constraint({i: -1, i + 1: 1 + r, i + 2: -r}, high=0)
        for i in range(1, count):
            t = npos + i - 1
            constraint({i: 1, t: -1}, high=normalized[i])
            constraint({i: -1, t: -1}, high=-normalized[i])
        continuous_rows = len(lower)
        for a, options in zip(free, choices):
            constraint({z: 1 for _, z in options}, 1, 1)
            for i, z in options:
                constraint({i: 1, z: 1}, high=1 + a)
                constraint({i: -1, z: 1}, high=1 - a)
        for prev, nxt in zip(choices[:-1], choices[1:]):
            constraint({**{z: -i for i, z in prev}, **{z: i for i, z in nxt}}, low=1)
        matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), nv)).tocsc()
        result = milp(
            cost,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0},
        )
        if result.status == 2:
            raise MeshInfeasibleError(
                "No mesh satisfies anchors, fixed collars, budget and grading/spacing"
            )
        if not result.success:
            raise MeshOptimizationError(f"Mesh optimization did not finish: {result.message}")
        # Refine the chosen integer assignment with exact anchor bounds and tighter
        # LP tolerances; snapping MILP coordinates alone can break ratio constraints.
        lp_lb, lp_ub = lb[: npos + ndev].copy(), ub[: npos + ndev].copy()
        assignments = []
        for a, options in zip(free, choices):
            index = max(options, key=lambda pair: result.x[pair[1]])[0]
            lp_lb[index] = lp_ub[index] = a
            assignments.append((index, min(anchors, key=lambda value: abs(value / length - a))))
        linear = matrix[:continuous_rows, : npos + ndev]
        lows, highs = np.array(lower[:continuous_rows]), np.array(upper[:continuous_rows])
        has_low, has_high = np.isfinite(lows), np.isfinite(highs)
        polished = linprog(
            cost[: npos + ndev],
            A_ub=vstack([linear[has_high], -linear[has_low]]),
            b_ub=np.r_[highs[has_high], -lows[has_low]],
            bounds=np.column_stack((lp_lb, lp_ub)),
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
        if not polished.success:
            raise MeshOptimizationError("Numerical refinement of anchor assignment failed")
        out = polished.x[:npos] * length
        for index, value in fixed.items():
            out[index] = value
        for index, value in assignments:
            out[index] = value
        gap = float(getattr(result, "mip_gap", 0.0) or 0.0)
        status = "optimal"
    try:
        validate_spacing(out, c)
    except MeshInfeasibleError as error:
        raise MeshOptimizationError("Optimizer output failed final spacing verification") from error
    if len(out) != count + 1 or not all(a in out for a in anchors):
        raise MeshOptimizationError("Final mesh failed anchor/count verification")
    interior_delta = abs(
        out[collar.cells + 1 : count - collar.cells]
        - target[collar.cells + 1 : count - collar.cells]
    )
    report = {
        "projection_l1": float(interior_delta.mean() / length) if interior_delta.size else 0.0,
        "projection_max": float(np.max(abs(out - target)) / length),
        "meshing_seconds": perf_counter() - started,
        "projection_status": status,
        "mip_gap": gap,
        "max_ratio_limit": c.max_ratio,
        "interior_cells": learned_count,
    }
    return (out, report) if return_diagnostics else out
