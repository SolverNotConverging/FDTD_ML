"""Sampled Ez dual-cell material averages; PEC boundaries retain point treatment.

Only nonmagnetic, isotropic 2D materials are supported. Ez is tangential to every
in-plane dielectric boundary. The geometry is sampled with last-primitive-wins
priority; epsilon and electric conductivity are averaged before coefficients.
"""

from types import SimpleNamespace

import numpy as np


def validate_averaging(mode, samples, max_samples, tolerance):
    if mode not in ("point", "sampled"):
        raise ValueError("material_averaging must be point or sampled")
    for value in (samples, max_samples):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError("Averaging sample counts must be integers >= 2")
    if max_samples > 256:
        raise ValueError("Averaging sample counts are capped at 256 per axis")
    if (
        max_samples < samples
        or max_samples % samples
        or (max_samples // samples) & ((max_samples // samples) - 1)
    ):
        raise ValueError("Averaging maximum must be the initial count times a power of two")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Averaging tolerance must be positive and finite")


def _dual_edges(nodes):
    return np.r_[nodes[0], (nodes[:-1] + nodes[1:]) / 2, nodes[-1]]


def _segment_mask(mask, xe, ye, a, b):
    """Mark dual rectangles touched by a segment (slab intersection test)."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ix0 = max(0, np.searchsorted(xe, lo[0], side="left") - 1)
    ix1 = min(len(xe) - 1, np.searchsorted(xe, hi[0], side="right"))
    iy0 = max(0, np.searchsorted(ye, lo[1], side="left") - 1)
    iy1 = min(len(ye) - 1, np.searchsorted(ye, hi[1], side="right"))
    if ix1 <= ix0 or iy1 <= iy0:
        return
    shape = (ix1 - ix0, iy1 - iy0)
    enter, leave = np.zeros(shape), np.ones(shape)
    lower = (xe[ix0:ix1, None], ye[None, iy0:iy1])
    upper = (xe[ix0 + 1 : ix1 + 1, None], ye[None, iy0 + 1 : iy1 + 1])
    for axis in range(2):
        delta = b[axis] - a[axis]
        if delta == 0:
            leave = np.where((a[axis] >= lower[axis]) & (a[axis] <= upper[axis]), leave, -1)
        else:
            t0 = (lower[axis] - a[axis]) / delta
            t1 = (upper[axis] - a[axis]) / delta
            enter = np.maximum(enter, np.minimum(t0, t1))
            leave = np.minimum(leave, np.maximum(t0, t1))
    mask[ix0:ix1, iy0:iy1] |= enter <= leave


def _boundary_mask(primitive, xe, ye, shape):
    kind, _, data = primitive
    mask = np.zeros(shape, dtype=bool)
    if kind == "circle":
        cx, cy, r = data
        nearx = np.maximum(np.maximum(xe[:-1] - cx, cx - xe[1:]), 0)
        neary = np.maximum(np.maximum(ye[:-1] - cy, cy - ye[1:]), 0)
        farx = np.maximum(abs(xe[:-1] - cx), abs(xe[1:] - cx))
        fary = np.maximum(abs(ye[:-1] - cy), abs(ye[1:] - cy))
        mask = (nearx[:, None] ** 2 + neary[None, :] ** 2 <= r * r) & (
            farx[:, None] ** 2 + fary[None, :] ** 2 >= r * r
        )
    elif kind == "polygon":
        for a, b in zip(data, np.roll(data, -1, axis=0)):
            _segment_mask(mask, xe, ye, a, b)
    else:
        xx, yy = np.broadcast_arrays(np.atleast_1d(data.x), np.atleast_1d(data.y))
        _segment_mask(mask, xe, ye, (xx[0], yy[0]), (xx[-1], yy[-1]))
    return mask


def _sample_pairs(scene, x, y):
    """Material values at paired points, unlike Scene.sample's Cartesian product."""
    eps = np.ones(x.shape)
    sigma = np.zeros(x.shape)
    pec = np.zeros(x.shape, dtype=bool)
    tol = 1e-12 * max(scene.Lx, scene.Ly)
    for kind, material, data in scene.primitives:
        if kind == "circle":
            cx, cy, r = data
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
        elif kind == "polygon":
            mask = np.zeros(x.shape, dtype=bool)
            boundary = mask.copy()
            for (xa, ya), (xb, yb) in zip(data, np.roll(data, -1, axis=0)):
                cross = (x - xa) * (yb - ya) - (y - ya) * (xb - xa)
                boundary |= (
                    (abs(cross) <= tol * np.hypot(xb - xa, yb - ya))
                    & (x >= min(xa, xb) - tol)
                    & (x <= max(xa, xb) + tol)
                    & (y >= min(ya, yb) - tol)
                    & (y <= max(ya, yb) + tol)
                )
                if ya != yb:
                    mask ^= ((ya > y) != (yb > y)) & (x < (xb - xa) * (y - ya) / (yb - ya) + xa)
            mask |= boundary
        else:
            # Zero-area PEC lines are protected separately, never volume averaged.
            continue
        eps[mask] = material.epsilon_r
        sigma[mask] = material.sigma_e
        pec[mask] = material.kind == "PEC"
    return eps, sigma, pec


def average_electric(scene, mesh, eps, sigma, pec, *, samples=8, max_samples=32, tolerance=1e-3):
    validate_averaging("sampled", samples, max_samples, tolerance)
    if any(m.kind != "PEC" and (m.mu_r != 1 or m.sigma_h != 0) for m in scene.materials.values()):
        raise ValueError("Sampled electric averaging currently requires mu_r=1 and sigma_h=0")
    xe, ye = _dual_edges(mesh.x), _dual_edges(mesh.y)
    candidates = np.zeros(eps.shape, dtype=bool)
    protected = pec.copy()
    for primitive in scene.primitives:
        boundary = _boundary_mask(primitive, xe, ye, eps.shape)
        if primitive[1].kind == "PEC":
            protected |= boundary
        else:
            candidates |= boundary
    candidates &= ~protected
    # An ordinary object can overwrite part of a PEC body. Its edge then forms
    # a PEC interface even though it is not an edge of the original PEC shape.
    # Conservatively protect the union of raw PEC volumes as well as their edges.
    ci, cj = np.nonzero(candidates)
    pec_geometry = [p for p in scene.primitives if p[1].kind == "PEC"]
    if pec_geometry and len(ci):
        raw_pec = SimpleNamespace(Lx=scene.Lx, Ly=scene.Ly, primitives=pec_geometry)
        _, _, inside = _sample_pairs(raw_pec, mesh.x[ci], mesh.y[cj])
        protected[ci[inside], cj[inside]] = True
        candidates &= ~protected
    ii, jj = np.nonzero(candidates)
    averaged_eps, averaged_sigma = eps.copy(), sigma.copy()
    counts = {}
    unresolved = 0
    unexpected_pec = 0
    # Bound temporary quadrature arrays even for fine reference grids.
    chunk = max(1, 262144 // max_samples**2)
    for first in range(0, len(ii), chunk):
        i, j = ii[first : first + chunk], jj[first : first + chunk]
        previous = None
        n = samples
        while len(i):
            u = (np.arange(n) + 0.5) / n
            px = xe[i, None, None] + (xe[i + 1] - xe[i])[:, None, None] * u[None, :, None]
            py = ye[j, None, None] + (ye[j + 1] - ye[j])[:, None, None] * u[None, None, :]
            x, y = np.broadcast_arrays(px, py)
            e, s, p = _sample_pairs(scene, x, y)
            current = np.column_stack((e.mean(axis=(1, 2)), s.mean(axis=(1, 2))))
            blocked = p.any(axis=(1, 2))
            # Defensive fallback if an unresolved/overlapping PEC fragment is found.
            unexpected_pec += int(blocked.sum())
            done = np.zeros(len(i), dtype=bool)
            if previous is not None:
                scale = np.maximum(abs(current), [1.0, 1e-12])
                done = np.all(abs(current - previous) <= tolerance * scale, axis=1)
            if n == max_samples:
                if previous is not None:
                    unresolved += int((~done & ~blocked).sum())
                done[:] = True
            accept = done & ~blocked
            averaged_eps[i[accept], j[accept]] = current[accept, 0]
            averaged_sigma[i[accept], j[accept]] = current[accept, 1]
            counts[n] = counts.get(n, 0) + int(accept.sum())
            keep = ~(done | blocked)
            i, j = i[keep], j[keep]
            previous = current[keep]
            n *= 2
    diagnostics = dict(
        mode="sampled",
        initial_samples=samples,
        max_samples=max_samples,
        relative_tolerance=tolerance,
        candidate_cells=len(ii),
        pec_protected_cells=int(protected.sum()),
        sampled_cells_by_axis_count=counts,
        at_sampling_limit=unresolved,
        unexpected_pec_fallback_cells=unexpected_pec,
        sampling_convergence_checked=max_samples > samples,
    )
    return averaged_eps, averaged_sigma, diagnostics
