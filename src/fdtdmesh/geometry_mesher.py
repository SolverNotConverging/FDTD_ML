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
    constraints = constraints or AxisConstraints(max_spacing=max(exterior_h, target_spacing))
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
    )
    started = perf_counter()
    previous_anchors = None
    assignment = "local"

    def fail(message):
        report["elapsed_seconds"] = perf_counter() - started
        raise GeometryMeshingError(message, report)

    for step in range(max_passes):
        if np.any(counts > caps):
            fail(
                f"Geometry-aware construction reached max_cells={tuple(map(int, caps))}; proposed {tuple(map(int, counts))}"
            )
        remaining = time_limit - (perf_counter() - started)
        if remaining <= 0:
            fail("Geometry-aware construction reached its time limit")
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
                fail(
                    "Fixed-budget geometry construction could not find a valid allocation; budget was not increased"
                )
            counts = reserve + np.maximum(
                counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
            )
            continue
        except MeshOptimizationError as exc:
            fail(f"Geometry-aware projection did not complete: {exc}")
        assignment = "local"
        issues, additions = inspect_mesh(geometry, mesh)
        report["issues"] = issues
        record.update(
            status="repair" if issues else "validating",
            violations=len(issues),
            topology_violations=sum(i["kind"] == "multiple_crossings" for i in issues),
            donor_violations=sum(i["kind"] == "unavailable_donor" for i in issues),
        )
        if not issues or (boundary is not None and boundary.mode == "hybrid"):
            try:
                build_conformal(geometry.to_scene(), mesh, boundary=boundary)
            except UnresolvedGeometryError as exc:
                record.update(status="conformal_rejection", message=str(exc))
                if cells is not None:
                    fail("Fixed-budget conformal preparation failed")
                counts = reserve + np.maximum(
                    counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
                )
                continue
            record["status"] = "valid"
            report.update(
                elapsed_seconds=perf_counter() - started,
                status="valid",
                exterior_fixed=True,
                boundary_mode="conformal" if boundary is None else boundary.mode,
                # Retain the successful witness coordinates for subsequent
                # fixed-budget density studies. They identify local material
                # intervals and donors that the repair loop needed to resolve.
                witness_anchors=[a.tolist() for a in anchors],
                geometry_sha256=hashlib.sha256(json_text(geometry.as_dict()).encode()).hexdigest(),
            )
            mesh.metadata.update(strategy="geometry_aware", geometry_aware=report)
            return mesh
        before = [len(a) for a in anchors]
        previous_anchors = anchors
        anchors = [
            _unique(np.r_[a, extra], length)
            for a, extra, length in zip(anchors, additions, geometry.size)
        ]
        if before == [len(a) for a in anchors]:
            if cells is not None:
                fail(
                    "Fixed-budget geometry construction could not find a valid allocation; budget was not increased"
                )
            counts = reserve + np.maximum(
                counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
            )
    fail(
        "Geometry-aware construction reached max_passes; inspect exception.report for unresolved edges"
    )
