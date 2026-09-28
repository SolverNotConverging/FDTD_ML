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


def _intervals(scene, axis, fixed, tol):
    """Merge adjacent vacuum pieces left by hidden CSG boundaries."""
    merged = []
    for lo, hi in scene.vacuum_intervals(axis, fixed):
        if hi - lo <= tol:
            continue
        if merged and lo <= merged[-1][1] + tol:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    return merged


def inspect_mesh(geometry, mesh):
    """Return exact edge-topology and donor violations with repair coordinates."""
    scene = geometry.to_scene()
    tol = 1e-11 * max(geometry.size)
    pec = scene.contains(mesh.x[:, None], mesh.y[None, :])
    issues, requests = [], [[], []]
    for axis in (0, 1):
        lines, fixed = (mesh.x, mesh.y) if axis == 0 else (mesh.y, mesh.x)
        mask = pec if axis == 0 else pec.T
        full = np.diff(lines)
        for j, value in enumerate(fixed):
            air = _intervals(scene, axis, value, tol)
            lengths = np.zeros(len(full))
            for lo, hi in air:
                lengths += np.maximum(0, np.minimum(lines[1:], hi) - np.maximum(lines[:-1], lo))
            cuts = np.unique([p for ab in air for p in ab if tol < p < lines[-1] - tol])
            # More than one transition is unsupported even with different endpoints.
            # cuts are sorted and unique; count open-interval intersections
            # without a Python loop over every Yee edge on every scanline.
            crossings = np.maximum(
                0,
                np.searchsorted(cuts, lines[1:] - tol, side="left")
                - np.searchsorted(cuts, lines[:-1] + tol, side="right"),
            )
            same_air = ~mask[:-1, j] & ~mask[1:, j]
            same_pec = mask[:-1, j] & mask[1:, j]
            bad = np.flatnonzero(
                (
                    (crossings > 1)
                    | (same_air & (lengths < full - tol))
                    | (same_pec & (lengths > tol))
                )
            )
            for i in bad:
                points = np.r_[
                    lines[i], cuts[(cuts > lines[i]) & (cuts < lines[i + 1])], lines[i + 1]
                ]
                issues.append(
                    dict(
                        kind="multiple_crossings",
                        axis=axis,
                        fixed=float(value),
                        edge=[float(lines[i]), float(lines[i + 1])],
                    )
                )
                for lo, hi in zip(points[:-1], points[1:]):
                    midpoint = (lo + hi) / 2
                    inside = (
                        scene.contains(midpoint, value)
                        if axis == 0
                        else scene.contains(value, midpoint)
                    )
                    missing = mask[i, j] == mask[i + 1, j] and inside != mask[i, j]
                    internal = lo > lines[i] + tol and hi < lines[i + 1] - tol
                    if hi - lo > tol and (missing or internal):
                        requests[axis].append(midpoint)
            if len(bad):
                continue  # Donors are meaningful only after this scanline's topology is resolved.
            all_air = same_air & (lengths >= full - tol)
            lengths[same_pec], lengths[all_air] = 0, full[all_air]
            used = set()
            for i in np.flatnonzero((lengths > 0) & (lengths < 0.5 * full * (1 - 1e-12))):
                borrow = 0.5 * full[i] - lengths[i]
                candidates = [i + (1 if mask[i, j] else -1)]
                donor = next(
                    (
                        d
                        for d in candidates
                        if 0 <= d < len(full)
                        and d not in used
                        and i not in used
                        and np.isclose(lengths[d], full[d], rtol=1e-11, atol=0)
                        and 0 < borrow < full[d]
                    ),
                    None,
                )
                if donor is not None:
                    used.update((i, donor))
                    continue
                issues.append(
                    dict(
                        kind="unavailable_donor",
                        axis=axis,
                        fixed=float(value),
                        edge=[float(lines[i]), float(lines[i + 1])],
                        open_length=float(lengths[i]),
                        donor_indices=list(map(int, candidates)),
                    )
                )
                for lo, hi in air:
                    if min(hi, lines[i + 1]) - max(lo, lines[i]) > tol:
                        # Two distinct complete air edges can serve the two cut ends.
                        requests[axis].extend(lo + (hi - lo) * np.array([0.25, 0.5, 0.75]))
                        break
    return issues, tuple(np.unique(a) for a in requests)


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
        raise ValueError("geometry_aware requires automatic domain or fit_domain()")
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
    counts = reserve + np.maximum(4, np.ceil(spans / target).astype(int))
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
        record = dict(cells=counts.tolist(), anchors=[len(a) for a in anchors])
        report["passes"].append(record)
        try:
            mesh = generate_mesh(
                geometry,
                layout,
                pml,
                tuple(counts),
                "uniform",
                anchors=anchors,
                anchor_assignment="local",
                constraints=constraints,
                time_limit=max(0.01, remaining / 2),
            )
        except MeshInfeasibleError as exc:
            record.update(status="projection_infeasible", message=str(exc))
            if previous_anchors is not None:
                # Repairs are proposals, not permanent corner constraints. If they
                # conflict with fixed exterior lines, retry a finer grid instead.
                anchors, previous_anchors = previous_anchors, None
            counts = reserve + np.maximum(
                counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
            )
            continue
        except MeshOptimizationError as exc:
            fail(f"Geometry-aware projection did not complete: {exc}")
        issues, additions = inspect_mesh(geometry, mesh)
        report["issues"] = issues
        record.update(
            status="repair" if issues else "validating",
            violations=len(issues),
            topology_violations=sum(i["kind"] == "multiple_crossings" for i in issues),
            donor_violations=sum(i["kind"] == "unavailable_donor" for i in issues),
        )
        if not issues:
            try:
                build_conformal(geometry.to_scene(), mesh)
            except UnresolvedGeometryError as exc:
                record.update(status="conformal_rejection", message=str(exc))
                counts = reserve + np.maximum(
                    counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
                )
                continue
            record["status"] = "valid"
            report.update(
                elapsed_seconds=perf_counter() - started,
                status="valid",
                exterior_fixed=True,
                # Retain the successful witness coordinates for subsequent
                # fixed-budget density studies. They identify local material
                # intervals and donors that the repair loop needed to resolve.
                witness_anchors=[a.tolist() for a in anchors],
                geometry_sha256=hashlib.sha256(
                    json_text(geometry.as_dict()).encode()
                ).hexdigest(),
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
            counts = reserve + np.maximum(
                counts - reserve + 4, np.ceil((counts - reserve) * 1.2).astype(int)
            )
    fail(
        "Geometry-aware construction reached max_passes; inspect exception.report for unresolved edges"
    )
