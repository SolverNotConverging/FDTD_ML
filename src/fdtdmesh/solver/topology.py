"""Shared exact scanline topology and enlarged-donor classification."""

import numpy as np


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


def inspect_scene(scene, mesh):
    """Return exact edge-topology and donor violations with repair coordinates."""
    tol = 1e-11 * max(scene.Lx, scene.Ly)
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
                        index=[int(i), int(j)],
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
                        index=[int(i), int(j)],
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
    # Between boundary extrema/intersections the ordering of analytic CSG
    # intervals is unchanged. These probes detect islands/holes completely
    # enclosed by a cell, which grid-edge intersections alone cannot see.
    from ..geometry_bounds import boundary_candidates

    points = boundary_candidates(scene)
    if len(points):
        # Cells already diagnosed by a grid-edge crossing use that edge's
        # targeted repair coordinates; do not overconstrain them with redundant
        # interior probes intended for completely enclosed features.
        covered = set()
        for issue in issues:
            i, j = issue["index"]
            if issue["axis"] == 0:
                covered.update(((i, j - 1), (i, j)))
            else:
                covered.update(((j - 1, i), (j, i)))
        levels = np.unique(np.r_[mesh.y, points[:, 1]])
        probes = (levels[:-1] + levels[1:]) / 2
        found = set()
        for y in probes[(probes > 0) & (probes < scene.Ly)]:
            j = int(np.searchsorted(mesh.y, y) - 1)
            vacuum = np.zeros(mesh.Nx)
            for lo, hi in _intervals(scene, 0, y, tol):
                vacuum += np.maximum(0, np.minimum(mesh.x[1:], hi) - np.maximum(mesh.x[:-1], lo))
            corners = pec[:-1, j].astype(int) + pec[1:, j] + pec[:-1, j + 1] + pec[1:, j + 1]
            bad = ((corners == 0) & (vacuum < np.diff(mesh.x) - tol)) | (
                (corners == 4) & (vacuum > tol)
            )
            for i in np.flatnonzero(bad):
                if (int(i), j) in found or (int(i), j) in covered:
                    continue
                found.add((int(i), j))
                issues.append(dict(kind="unresolved_cell_feature", cell=[int(i), j]))
                local = points[
                    (points[:, 0] >= mesh.x[i] - tol)
                    & (points[:, 0] <= mesh.x[i + 1] + tol)
                    & (points[:, 1] >= mesh.y[j] - tol)
                    & (points[:, 1] <= mesh.y[j + 1] + tol)
                ]
                requests[0].extend(local[:, 0])
                requests[1].extend(local[:, 1])
                requests[1].append(float(y))
                for lo, hi in _intervals(scene, 0, y, tol):
                    for v in (lo, hi, (lo + hi) / 2):
                        if mesh.x[i] < v < mesh.x[i + 1]:
                            requests[0].append(float(v))
    return issues, tuple(np.unique(a) for a in requests)
