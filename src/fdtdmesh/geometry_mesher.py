"""Deterministic exact-CSG mesh construction with topology and donor repairs.

All repairs move/add tensor-product lines, never modify the material recipe.
Only meshes accepted by the actual conformal enlarged-cell operator are returned.
"""

import hashlib
from time import perf_counter

import numpy as np

from .mesh import AxisConstraints, MeshInfeasibleError, MeshOptimizationError, cell_count
from .result import json_text
from .solver.conformal import UnresolvedGeometryError, build_conformal
from .strategies import generate_mesh


class GeometryMeshingError(MeshInfeasibleError):
    """Construction exhausted its limits; report contains the unresolved locations."""

    def __init__(self, message, report):
        self.report = report
        super().__init__(message)


def inspect_mesh(geometry, mesh):
    from .solver.topology import inspect_scene

    return inspect_scene(geometry.to_scene(), mesh)


def _unique(values, scale):
    values = np.sort(np.asarray(values, float))
    return (
        values[np.r_[True, np.diff(values) > 64 * np.finfo(float).eps * scale]]
        if len(values)
        else values
    )


def geometry_aware_mesh(
    geometry,
    layout,
    pml,
    *,
    target_spacing,
    constraints=None,
    max_cells=(512, 512),
    max_passes=20,
    time_limit=30.0,
    cells=None,
    boundary=None,
    hybrid_repair_passes=3,
):
    """Construct within resource caps; target_spacing is a target, not a cell budget.

    Requires a fitted layout. Fixed exterior coordinates never move. Time limits
    are checked between geometry passes and passed to the axis projectors.
    """
    if not np.isfinite(target_spacing) or target_spacing <= 0:
        raise ValueError("target_spacing must be positive and finite")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    max_passes = cell_count(max_passes)
    if (
        isinstance(hybrid_repair_passes, bool)
        or not np.isfinite(hybrid_repair_passes)
        or int(hybrid_repair_passes) != hybrid_repair_passes
        or hybrid_repair_passes < 0
    ):
        raise ValueError("hybrid_repair_passes must be a nonnegative integer")
    hybrid = boundary is not None and boundary.mode == "hybrid"
    if len(max_cells) != 2:
        raise ValueError("max_cells must contain two axis cell caps")
    caps = np.array([cell_count(n) for n in max_cells])
    if layout.scatterer_margin_cells is None or layout.exterior_cells is None:
        raise ValueError("geometry_aware requires a resolved automatic-domain layout")
    bounds = geometry.bounds
    if bounds is None:
        raise ValueError("geometry_aware requires nonempty PEC geometry")
    spans = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2]])
    reserve = np.array(
        [
            2 * (p.cells + sum(layout.exterior_cells) + layout.scatterer_margin_cells)
            for p in (pml.x, pml.y)
        ]
    )
    exterior_h = max(p.thickness / p.cells for p in (pml.x, pml.y) if p.cells)
    constraints = constraints or AxisConstraints(
        max_spacing=max(exterior_h * (1.4 if layout.margin_mesh == "graded" else 1), target_spacing)
    )
    if constraints.max_spacing is not None and constraints.max_spacing < exterior_h * (1 - 1e-10):
        raise MeshInfeasibleError("max_spacing is smaller than the fixed exterior spacing")
    target = min(target_spacing, constraints.max_spacing or target_spacing)
    counts = (
        reserve + np.maximum(4, np.ceil(spans / target).astype(int))
        if cells is None
        else np.array([cell_count(n) for n in cells])
    )
    if counts.shape != (2,) or np.any(counts <= reserve):
        raise MeshInfeasibleError(f"Budget must exceed reserved axis cells {tuple(reserve)}")
    if cells is not None:
        caps = counts.copy()
    anchors = [np.array([]), np.array([])]
    # Probe each primitive in both directions, including air cutters. This seeds
    # interior features that could otherwise lie entirely inside one grid cell.
    for x0, x1, y0, y1 in geometry.to_scene().bounds():
        for axis, coordinate in enumerate(((x0 + x1) / 2, (y0 + y1) / 2)):
            lo, hi = bounds[2 * axis : 2 * axis + 2]
            if lo < coordinate < hi:
                anchors[axis] = np.r_[anchors[axis], coordinate]
    report = dict(
        strategy="geometry_aware",
        target_spacing=float(target_spacing),
        max_cells=caps.tolist(),
        max_passes=max_passes,
        passes=[],
        issues=[],
        hybrid_repair_passes=hybrid_repair_passes,
    )
    started = perf_counter()
    previous_anchors = None
    assignment = "local"
    best = None
    best_score = None
    hybrid_attempts = stagnant = 0

    def finish(selected, reason):
        candidate, selected_anchors, selected_issues, details = selected
        report.update(
            elapsed_seconds=perf_counter() - started,
            status="valid",
            exterior_fixed=True,
            margin_mesh=layout.margin_mesh,
            boundary_mode="conformal" if boundary is None else boundary.mode,
            witness_anchors=[a.tolist() for a in selected_anchors],
            issues=selected_issues,
            boundary=details,
            termination_reason=reason,
            geometry_sha256=hashlib.sha256(json_text(geometry.as_dict()).encode()).hexdigest(),
        )
        candidate.metadata.update(strategy="geometry_aware", geometry_aware=report)
        return candidate

    def fail(message):
        if best is not None:
            report["repair_limit_message"] = message
            return finish(best, "retained_best_hybrid")
        report["elapsed_seconds"] = perf_counter() - started
        raise GeometryMeshingError(message, report)

    for step in range(max_passes):
        if np.any(counts > caps):
            return fail(
                f"Geometry-aware construction reached max_cells={tuple(map(int, caps))}; proposed {tuple(map(int, counts))}"
            )
        remaining = time_limit - (perf_counter() - started)
        if remaining <= 0:
            return fail("Geometry-aware construction reached its time limit")
        anchors = [_unique(a, length) for a, length in zip(anchors, geometry.size)]
        record = dict(
            cells=counts.tolist(), anchors=[len(a) for a in anchors], assignment=assignment
        )
        report["passes"].append(record)
        try:
            mesh = generate_mesh(
                geometry,
                layout,
                pml,
                tuple(counts),
                "uniform",
                anchors=anchors,
                anchor_assignment=assignment,
                constraints=constraints,
                time_limit=max(0.01, remaining / 2),
            )
        except MeshInfeasibleError as exc:
            record.update(status="projection_infeasible", message=str(exc))
            if assignment == "local":
                assignment = "joint"
                continue
            assignment = "local"
            if previous_anchors is not None:
                # Repairs are proposals, not permanent corner constraints. If they
                # conflict with fixed exterior lines, retry a finer grid instead.
                anchors, previous_anchors = previous_anchors, None
            if cells is not None:
                return fail(
                    "Fixed-budget geometry construction could not find a valid allocation; budget was not increased"
                )
            counts = reserve + np.maximum(
                counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
            )
            continue
        except MeshOptimizationError as exc:
            record.update(status="projection_incomplete", message=str(exc))
            return fail(f"Geometry-aware projection did not complete: {exc}")
        assignment = "local"
        issues, additions = inspect_mesh(geometry, mesh)
        report["issues"] = issues
        record.update(
            status="repair" if issues else "validating",
            violations=len(issues),
            topology_violations=sum(i["kind"] == "multiple_crossings" for i in issues),
            donor_violations=sum(i["kind"] == "unavailable_donor" for i in issues),
        )
        if not issues or hybrid:
            details = {}
            try:
                build_conformal(geometry.to_scene(), mesh, boundary=boundary, report=details)
            except UnresolvedGeometryError as exc:
                record.update(status="conformal_rejection", message=str(exc))
                if not hybrid:
                    if cells is not None:
                        return fail("Fixed-budget conformal preparation failed")
                    counts = reserve + np.maximum(
                        counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
                    )
                    continue
            else:
                selected = (mesh, [a.copy() for a in anchors], issues, details)
                record.update(
                    status="valid",
                    fallback_cells=details["fallback_cells"],
                    fallback_area_m2=details["fallback_area_m2"],
                    fallback_max_diameter_m=details["fallback_max_diameter_m"],
                )
                if not details["fallback_cells"]:
                    return finish(selected, "fully_conformal")
                # Physical patch extent, not cell count, ranks preparation attempts.
                score = (details["fallback_max_diameter_m"], details["fallback_area_m2"])
                improved = best_score is None or score < best_score
                if improved:
                    best, best_score = selected, score
                stagnant = 0 if improved else stagnant + 1
                if stagnant >= 2:
                    return finish(best, "persistent_fallback")
        if hybrid:
            hybrid_attempts += 1
            if hybrid_attempts >= 1 + hybrid_repair_passes:
                if best is not None:
                    return finish(best, "hybrid_repair_limit")
                return fail("Hybrid repair limit reached without a policy-accepted mesh")
        before = [len(a) for a in anchors]
        previous_anchors = anchors
        anchors = [
            _unique(np.r_[a, extra], length)
            for a, extra, length in zip(anchors, additions, geometry.size)
        ]
        if before == [len(a) for a in anchors]:
            if best is not None:
                return finish(best, "persistent_fallback")
            if cells is not None:
                return fail(
                    "Fixed-budget geometry construction could not find a valid allocation; budget was not increased"
                )
            counts = reserve + np.maximum(
                counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
            )
    return fail(
        "Geometry-aware construction reached max_passes; inspect exception.report for unresolved edges"
    )
