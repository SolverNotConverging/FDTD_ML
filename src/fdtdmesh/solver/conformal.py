"""TMz contour-path Faraday operators with conservative enlarged-cell pairs.

Invariance along z makes a magnetic cut-face area its open x/y length times
unit z depth. Small faces borrow vacuum area; geometry is never inflated.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags


class UnresolvedGeometryError(ValueError):
    """The grid cannot represent the geometry without changing its topology."""


@dataclass
class FaradayOperator:
    length: np.ndarray
    diagonal: np.ndarray
    peer: np.ndarray
    coupling: np.ndarray
    pairs: list


def open_lengths(scene, mesh, axis, pec, *, validate=True):
    lines, fixed = (mesh.x, mesh.y) if axis == 0 else (mesh.y, mesh.x)
    mask = pec if axis == 0 else pec.T
    lengths = np.zeros((len(lines) - 1, len(fixed)))
    tol = 1e-11 * max(scene.Lx, scene.Ly)
    for j, value in enumerate(fixed):
        for lo, hi in scene.vacuum_intervals(axis, value):
            lengths[:, j] += np.maximum(0, np.minimum(lines[1:], hi) - np.maximum(lines[:-1], lo))
        full = np.diff(lines)
        both_vacuum = ~mask[:-1, j] & ~mask[1:, j]
        both_metal = mask[:-1, j] & mask[1:, j]
        if validate and np.any(both_vacuum & (lengths[:, j] < full - tol)):
            raise UnresolvedGeometryError(
                "PEC crosses an edge with two vacuum endpoints; refine grid"
            )
        if validate and np.any(both_metal & (lengths[:, j] > tol)):
            raise UnresolvedGeometryError(
                "Vacuum gap crosses an edge with two PEC endpoints; refine grid"
            )
        lengths[both_metal, j] = 0
        lengths[both_vacuum, j] = full[both_vacuum]
    return lengths if axis == 0 else lengths.T


def enlarge(length, full, pec, axis, forbidden=None):
    """Pair a small face with a complete face on its vacuum side.

    Cut length s, donor d, borrowed length b, S=s+b, r=b/d:
      g_s = (f_s + r*f_d)/S; g_d = r*g_s + (1-r)*f_d/d.
    Then s*g_s+d*g_d=f_s+f_d and the inverse-length matrix is SPD.
    Pairs are disjoint; unavailable donors cause explicit rejection.
    """
    q = np.divide(1.0, length, out=np.zeros_like(length), where=length > 0)
    peer = np.full(length.shape, -1, dtype=np.int32)
    coupling = np.zeros_like(length)
    pairs = []
    full = np.broadcast_to(full, length.shape)
    small = np.argwhere((length > 0) & (length < 0.5 * full * (1 - 1e-12)))
    for i, j in small:
        step = 1 if pec[i, j] else -1
        ii, jj = (i + step, j) if axis == 0 else (i, j + step)
        if (
            not (0 <= ii < length.shape[0] and 0 <= jj < length.shape[1])
            or not np.isclose(length[ii, jj], full[ii, jj], rtol=1e-11, atol=0)
            or peer[ii, jj] != -1
            or (forbidden is not None and forbidden[ii, jj])
        ):
            raise UnresolvedGeometryError(
                "Small conformal face has no unused full vacuum donor; refine gap"
            )
        s, d = length[i, j], length[ii, jj]
        b = 0.5 * full[i, j] - s
        if not 0 < b < d:
            raise UnresolvedGeometryError("Nonuniform donor cannot supply the enlarged face")
        S, r = s + b, b / d
        q[i, j], q[ii, jj] = 1 / S, (1 - r) / d + r * r / S
        coupling[i, j] = coupling[ii, jj] = r / S
        a = np.ravel_multi_index((i, j), length.shape)
        z = np.ravel_multi_index((ii, jj), length.shape)
        peer[i, j], peer[ii, jj] = z, a
        pairs.append((int(a), int(z), float(b)))
    return FaradayOperator(length, q, peer, coupling, pairs)


def apply_operator(op, difference):
    out = op.diagonal * difference
    active = op.peer >= 0
    out[active] += op.coupling[active] * difference.ravel()[op.peer[active]]
    return out


def stiffness(mesh, pec, hx, hy):
    """K=sum B.T W Q B and M=dual_x*dual_y on the active E nodes.

    Transverse dual widths commute with each enlarged pair. K is symmetric
    positive semidefinite, providing an energy norm and a spectral CFL bound.
    """
    dx, dy = np.diff(mesh.x), np.diff(mesh.y)
    ux = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
    uy = np.r_[dy[0] / 2, (dy[:-1] + dy[1:]) / 2, dy[-1] / 2]
    active = np.flatnonzero(~pec.ravel())
    lookup = np.full(pec.size, -1)
    lookup[active] = np.arange(len(active))
    K = None
    for axis, op in ((0, hy), (1, hx)):
        i, j = np.indices(op.length.shape)
        left = i * (mesh.Ny + 1) + j
        right = left + (mesh.Ny + 1 if axis == 0 else 1)
        rows = np.tile(np.arange(left.size), 2)
        cols = lookup[np.r_[left.ravel(), right.ravel()]]
        values = np.r_[-np.ones(left.size), np.ones(left.size)]
        keep = cols >= 0
        B = coo_matrix(
            (values[keep], (rows[keep], cols[keep])), shape=(left.size, len(active))
        ).tocsr()
        transverse = np.broadcast_to(uy[None, :] if axis == 0 else ux[:, None], op.length.shape)
        p = op.peer.ravel()
        ix = np.flatnonzero(p >= 0)
        Q = diags((transverse * op.diagonal).ravel()) + coo_matrix(
            ((transverse * op.coupling).ravel()[ix], (ix, p[ix])), shape=(left.size, left.size)
        )
        term = B.T @ Q @ B
        K = term if K is None else K + term
    mass = (ux[:, None] * uy[None, :]).ravel()[active]
    return K.tocsr(), mass, active


def build_conformal(scene, mesh, *, boundary=None, report=None, warn=False):
    """Prepare strict or hybrid edges once; shared edges have a single operator."""
    import warnings

    from ..boundary import BoundaryPolicy, StaircaseFallbackWarning
    from .topology import inspect_scene

    policy = boundary or BoundaryPolicy()
    issues, _ = inspect_scene(scene, mesh)
    if issues and policy.mode == "conformal":
        mask = scene.contains(mesh.x[:, None], mesh.y[None, :])
        open_lengths(scene, mesh, 0, mask)
        open_lengths(scene, mesh, 1, mask)
        if any(i["kind"] == "unresolved_cell_feature" for i in issues):
            raise UnresolvedGeometryError(
                "Resolved feature lies entirely inside a cell or is unresolved by its nodes; refine grid"
            )
        raise UnresolvedGeometryError(
            f"Unresolved conformal geometry: {issues[0]['kind']}; refine grid"
        )
    pec = scene.contains(mesh.x[:, None], mesh.y[None, :])
    lx = open_lengths(scene, mesh, 0, pec, validate=False)
    ly = open_lengths(scene, mesh, 1, pec, validate=False)
    patch = np.zeros((mesh.Nx, mesh.Ny), bool)
    reasons = {}

    def mark(axis, i, j):
        if axis == 0:
            patch[i, max(0, j - 1) : min(mesh.Ny, j + 1)] = True
        else:
            patch[max(0, i - 1) : min(mesh.Nx, i + 1), j] = True

    for issue in issues:
        kind = issue["kind"]
        reasons[kind] = reasons.get(kind, 0) + 1
        if "cell" in issue:
            patch[tuple(issue["cell"])] = True
        else:
            i, j = issue["index"]
            mark(issue["axis"], i, j) if issue["axis"] == 0 else mark(1, j, i)
    initial = int(patch.sum())
    boundary_cells = patch.copy()
    corners = pec[:-1, :-1].astype(int) + pec[1:, :-1] + pec[:-1, 1:] + pec[1:, 1:]
    boundary_cells |= (corners > 0) & (corners < 4)
    # Closure terminates because each failed donor adds at least one cell to
    # the patch, and full staircasing has no small faces requiring enlargement.
    for _ in range(mesh.Nx * mesh.Ny + 1):
        fx = np.zeros_like(lx, dtype=bool)
        fy = np.zeros_like(ly, dtype=bool)
        fx[:, :-1] |= patch
        fx[:, 1:] |= patch
        fy[:-1, :] |= patch
        fy[1:, :] |= patch
        xlen, ylen = lx.copy(), ly.copy()
        xfull = np.broadcast_to(np.diff(mesh.x)[:, None], lx.shape)
        yfull = np.broadcast_to(np.diff(mesh.y)[None, :], ly.shape)
        xlen[fx] = np.where((pec[:-1] & pec[1:])[fx], 0, xfull[fx])
        ylen[fy] = np.where((pec[:, :-1] & pec[:, 1:])[fy], 0, yfull[fy])
        failed = []
        operators = []
        for axis, lengths, full, forbidden in ((0, xlen, xfull, fx), (1, ylen, yfull, fy)):
            # Rebuild the same disjoint donor allocation used by enlarge.
            used = set()
            for i, j in np.argwhere((lengths > 0) & (lengths < 0.5 * full * (1 - 1e-12))):
                step = 1 if pec[i, j] else -1
                ii, jj = (i + step, j) if axis == 0 else (i, j + step)
                valid = (
                    0 <= ii < lengths.shape[0]
                    and 0 <= jj < lengths.shape[1]
                    and not forbidden[ii, jj]
                    and (ii, jj) not in used
                    and (i, j) not in used
                    and np.isclose(lengths[ii, jj], full[ii, jj], rtol=1e-11, atol=0)
                    and 0 < 0.5 * full[i, j] - lengths[i, j] < full[ii, jj]
                )
                if valid:
                    used.update(((i, j), (ii, jj)))
                else:
                    failed.append((axis, int(i), int(j)))
            if not failed:
                operators.append(enlarge(lengths, full, pec, axis, forbidden))
        if not failed:
            hy, hx = operators
            break
        if policy.mode == "conformal":
            raise UnresolvedGeometryError("Small conformal face has no valid enlarged-cell donor")
        for axis, i, j in failed:
            mark(axis, i, j)
        reasons["donor_closure"] = reasons.get("donor_closure", 0) + len(failed)
    else:
        raise UnresolvedGeometryError("Hybrid patch closure failed")
    area = np.diff(mesh.x)[:, None] * np.diff(mesh.y)[None, :]
    fraction = float(area[patch].sum() / area.sum())
    from scipy.ndimage import find_objects, label

    labels, count = label(patch)  # Four-neighbour connectivity: shared edges.
    patches = []
    for number, slices in enumerate(find_objects(labels), start=1):
        local = labels[slices] == number
        sx, sy = slices
        bounds = [
            float(mesh.x[sx.start]),
            float(mesh.x[sx.stop]),
            float(mesh.y[sy.start]),
            float(mesh.y[sy.stop]),
        ]
        patches.append(
            dict(
                cells=int(local.sum()),
                area_m2=float(area[slices][local].sum()),
                bounds=bounds,
                # Bounding-box diagonal is a conservative physical diameter, not an error estimate.
                diameter_m=float(np.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2])),
            )
        )
    max_diameter = max((p["diameter_m"] for p in patches), default=0.0)
    details = dict(
        mode=policy.mode,
        fallback_cells=int(patch.sum()),
        initial_fallback_cells=initial,
        expanded_cells=int(patch.sum()) - initial,
        fallback_area_fraction=fraction,
        fallback_area_m2=float(area[patch].sum()),
        fallback_max_diameter_m=max_diameter,
        patch_count=count,
        patches=patches,
        reasons=reasons,
        boundary_cells=int(boundary_cells.sum()),
        fallback_boundary_fraction=float(
            (patch & boundary_cells).sum() / max(1, boundary_cells.sum())
        ),
        fallback_indices=np.argwhere(patch).tolist(),
        fallback_x_edges=np.argwhere(fx).tolist(),
        fallback_y_edges=np.argwhere(fy).tolist(),
        raw_issues=issues,
        enlarged_pairs=len(hx.pairs) + len(hy.pairs),
    )
    if patch.any() and (policy.on_fallback == "error" or fraction > policy.max_fallback_fraction):
        raise UnresolvedGeometryError(
            f"Staircase fallback exceeds policy: {int(patch.sum())} cells, area fraction {fraction:.6g}"
        )
    if policy.max_patch_diameter is not None and max_diameter > policy.max_patch_diameter:
        raise UnresolvedGeometryError(
            f"Staircase patch diameter {max_diameter:.6g} m exceeds max_patch_diameter"
        )
    if report is not None:
        report.update(details)
    if warn and patch.any():
        warnings.warn(
            f"Local staircase fallback in {int(patch.sum())} cells ({fraction:.3%} domain area); "
            "thin PEC/gaps may be lost. Inspect discretization diagnostics.",
            StaircaseFallbackWarning,
            stacklevel=2,
        )
    pec[[0, -1], :] = True
    pec[:, [0, -1]] = True
    K, mass, _ = stiffness(mesh, pec, hx, hy)
    bound = float(np.max(np.asarray(abs(K).sum(axis=1)).ravel() / mass))
    return pec, hx, hy, bound
