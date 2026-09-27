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


def open_lengths(scene, mesh, axis, pec):
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
        if np.any(both_vacuum & (lengths[:, j] < full - tol)):
            raise UnresolvedGeometryError(
                "PEC crosses an edge with two vacuum endpoints; refine grid"
            )
        if np.any(both_metal & (lengths[:, j] > tol)):
            raise UnresolvedGeometryError(
                "Vacuum gap crosses an edge with two PEC endpoints; refine grid"
            )
        lengths[both_metal, j] = 0
        lengths[both_vacuum, j] = full[both_vacuum]
    return lengths if axis == 0 else lengths.T


def enlarge(length, full, pec, axis):
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


def build_conformal(scene, mesh):
    for x0, x1, y0, y1 in scene.bounds():
        # Ordered air cutters may lie wholly outside a fitted domain. They have
        # no physical intersection with any cell and need no grid representation.
        if x1 <= 0 or x0 >= scene.Lx or y1 <= 0 or y0 >= scene.Ly:
            continue
        if not np.any((mesh.x >= x0) & (mesh.x <= x1)) and not np.any(
            (mesh.y >= y0) & (mesh.y <= y1)
        ):
            raise UnresolvedGeometryError("Primitive lies entirely inside a cell; refine grid")
    pec = scene.contains(mesh.x[:, None], mesh.y[None, :])
    lx, ly = open_lengths(scene, mesh, 0, pec), open_lengths(scene, mesh, 1, pec)
    hy = enlarge(lx, np.diff(mesh.x)[:, None], pec, 0)
    hx = enlarge(ly, np.diff(mesh.y)[None, :], pec, 1)
    pec[[0, -1], :] = True
    pec[:, [0, -1]] = True
    K, mass, _ = stiffness(mesh, pec, hx, hy)
    bound = float(np.max(np.asarray(abs(K).sum(axis=1)).ravel() / mass))
    return pec, hx, hy, bound
