"""Experimental TMz PEC cut-edge treatment on unchanged Cartesian Yee grids.

For z-invariant TMz, a cut vertical magnetic face reduces to an open x/y
interval. Its Faraday update uses the actual open length and total Ez=0 at
the PEC intersection. The electric dual metrics remain those of the Yee grid.
This is an explicit cut-edge specialization, not a general 3D cut-face solver.
Small intervals are retained and their time-step restriction is respected.
"""

import numpy as np
from scipy.sparse import coo_matrix

from .constants import C0
from .geometry import Circle, Rectangle
from .geometry_v2 import Ellipse, Polygon, PolygonWithHoles, SmoothLobed


def _intervals(objects, coordinate, axis, tolerance):
    """Union of PEC intervals along one coordinate line, with exact primitives."""
    intervals = []
    for obj in objects:
        if isinstance(obj, Circle):
            transverse = coordinate - obj.center[1 - axis]
            squared = obj.radius**2 - transverse**2
            if squared >= -tolerance * (2 * obj.radius + tolerance):
                half = np.sqrt(max(0.0, squared))
                intervals.append((obj.center[axis] - half, obj.center[axis] + half))
        elif isinstance(obj, Rectangle):
            bounds = obj.bounds
            if (
                bounds[2 * (1 - axis)] - tolerance
                <= coordinate
                <= bounds[2 * (1 - axis) + 1] + tolerance
            ):
                intervals.append((bounds[2 * axis], bounds[2 * axis + 1]))
        elif isinstance(obj, (Ellipse, Polygon, PolygonWithHoles, SmoothLobed)):
            intervals.extend(obj.line_intervals(coordinate, axis, tolerance))
        else:
            raise ValueError("Unsupported conformal PEC geometry")
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


class CutCellPEC:
    def __init__(self, grid, objects, mode="conformal"):
        objects = tuple(objects)
        if mode not in ("conformal", "staircase", "enlarged"):
            raise ValueError("PEC mode must be conformal, enlarged, or staircase")
        self.grid, self.mode = grid, mode
        self.coordinate_tolerance = 64 * np.finfo(float).eps * max(grid.x[-1], grid.y[-1])
        tolerance = self.coordinate_tolerance
        self.inside = np.zeros(grid.shape, dtype=bool)
        for obj in objects:
            if isinstance(obj, Rectangle):
                a, b, c, d = obj.bounds
                mask = (
                    (grid.x[:, None] >= a - tolerance)
                    & (grid.x[:, None] <= b + tolerance)
                    & (grid.y[None, :] >= c - tolerance)
                    & (grid.y[None, :] <= d + tolerance)
                )
            elif isinstance(obj, Circle):
                mask = (grid.x[:, None] - obj.center[0]) ** 2 + (
                    grid.y[None, :] - obj.center[1]
                ) ** 2 <= (obj.radius + tolerance) ** 2
            elif isinstance(obj, (Ellipse, Polygon, PolygonWithHoles, SmoothLobed)):
                mask = obj.contains(grid.x[:, None], grid.y[None, :])
            else:
                raise ValueError("Unsupported conformal PEC geometry")
            if not mask.any():
                raise ValueError(
                    "PEC has no represented Ez nodes; refine mesh or use a split-edge model"
                )
            self.inside |= mask
        self.inverse_lengths, self.boundaries = [], []
        fractions = []
        for axis, (nodes, transverse) in enumerate(((grid.x, grid.y), (grid.y, grid.x))):
            shape = (len(nodes) - 1, len(transverse))
            inverse = np.zeros(shape)
            boundary = []
            inside = self.inside if axis == 0 else self.inside.T
            for j, coordinate in enumerate(transverse):
                intervals = _intervals(objects, coordinate, axis, tolerance)
                # Snap roundoff-sized coordinate differences only, never a
                # finite cut fraction to make the time-step restriction vanish.
                intervals = [
                    (
                        nodes[np.argmin(abs(nodes - a))]
                        if np.min(abs(nodes - a)) <= tolerance
                        else a,
                        nodes[np.argmin(abs(nodes - b))]
                        if np.min(abs(nodes - b)) <= tolerance
                        else b,
                    )
                    for a, b in intervals
                ]
                for i, (left, right) in enumerate(zip(nodes[:-1], nodes[1:])):
                    low, high = inside[i, j], inside[i + 1, j]
                    if mode == "staircase":
                        if low and high:
                            continue
                        inverse[i, j] = 1 / (right - left)
                        if low != high:
                            hit = left if low else right
                        else:
                            continue
                    else:
                        # Positive-length pieces of PEC on this edge. A tangent
                        # contact matters only when it coincides with an Ez node.
                        hits = [
                            (max(left, a), min(right, b))
                            for a, b in intervals
                            if a <= right and b >= left
                        ]
                        occupied = sum(max(0.0, b - a) for a, b in hits)
                        if low and high:
                            if abs(occupied - (right - left)) > 1e-12 * (right - left):
                                raise ValueError(
                                    "Unresolved vacuum gap between PEC nodes; refine mesh"
                                )
                            continue
                        if not low and not high:
                            if occupied > 0:
                                raise ValueError(
                                    "PEC splits an edge with two exterior endpoints; refine mesh"
                                )
                            inverse[i, j] = 1 / (right - left)
                            continue
                        # Exactly one exterior endpoint; reject additional
                        # isolated PEC islands/gaps that require extra H unknowns.
                        hit = max(b for a, b in hits) if low else min(a for a, b in hits)
                        open_length = right - hit if low else hit - left
                        if abs(occupied + open_length - (right - left)) > 1e-12 * (right - left):
                            raise ValueError(
                                "Multiple PEC cuts on one edge require a split-edge model"
                            )
                        if open_length <= 0:
                            raise ValueError("Nonpositive open PEC edge")
                        inverse[i, j] = 1 / open_length
                        fractions.append(open_length / (right - left))
                    exterior = i + 1 if low else i
                    # sign for increasing-coordinate derivative.
                    sign = 1 if low else -1
                    if axis == 0:
                        boundary.append((i, j, exterior, j, hit, coordinate, sign))
                    else:
                        boundary.append((j, i, j, exterior, coordinate, hit, sign))
            self.inverse_lengths.append(inverse if axis == 0 else inverse.T)
            if boundary:
                b = np.asarray(boundary)
                self.boundaries.append(
                    (
                        b[:, 0].astype(int),
                        b[:, 1].astype(int),
                        b[:, 2].astype(int),
                        b[:, 3].astype(int),
                        b[:, 4],
                        b[:, 5],
                        b[:, 6],
                    )
                )
            else:
                self.boundaries.append(None)
        self.minimum_fraction = min(fractions, default=1.0)
        self.boundary_edge_count = sum(len(b[0]) for b in self.boundaries if b is not None)
        self.subcell_edge_count = int(sum(f < 1 - 1e-12 for f in fractions))
        self.enlargement = CellEnlargement(self) if mode == "enlarged" else None

    def gradients(self, ez, source, time):
        gradients = [
            np.diff(ez, axis=axis) * inverse for axis, inverse in enumerate(self.inverse_lengths)
        ]
        for gradient, inverse, boundary in zip(gradients, self.inverse_lengths, self.boundaries):
            if boundary is None:
                continue
            i, j, ei, ej, x, y, sign = boundary
            # At the physical PEC intersection Es=-Ei. The stored interior-node
            # field is not a substitute for this off-grid boundary value.
            gradient[i, j] = sign * (ez[ei, ej] + source.electric(x, y, time)) * inverse[i, j]
        return gradients

    def stable_time_step(self):
        """Conservative Gershgorin bound of the exterior electric wave operator.

        The operator is similar to a symmetric positive semidefinite matrix
        under Yee dual-area weights. Each absolute row sum is <=2*diagonal;
        leapfrog requires dt*sqrt(lambda_max)<=2. PEC only, vacuum exterior.
        """
        if self.enlargement is not None:
            return self.enlargement.stable_dt
        invx, invy = self.inverse_lengths
        dx, dy = np.diff(self.grid.x), np.diff(self.grid.y)
        dualx, dualy = (dx[:-1] + dx[1:]) / 2, (dy[:-1] + dy[1:]) / 2
        diagonal = (invx[:-1, 1:-1] + invx[1:, 1:-1]) / dualx[:, None] + (
            invy[1:-1, :-1] + invy[1:-1, 1:]
        ) / dualy[None, :]
        diagonal[self.inside[1:-1, 1:-1]] = 0
        return np.sqrt(2) / (C0 * np.sqrt(diagonal.max()))


class CellEnlargement:
    """Local algebraic cell aggregation for TMz, with a diagonal reduced mass.

    A near-boundary exterior Ez node is constrained by linear interpolation
    between the exact zero-total-E PEC intersection and a farther vacuum node.
    Both curl coupling and electric mass are projected with the same P:
    Mr=P.T M P, Kr=P.T K P. Merely enlarging a magnetic denominator would not
    be this method. This is an experimental 2D aggregation, not a reproduction
    of a particular published 3D face-borrowing implementation.
    """

    def __init__(self, cut, fraction_threshold=0.5):
        grid = cut.grid
        shape = grid.shape
        size = np.prod(shape)
        indices = np.arange(size).reshape(shape)
        free = ~cut.inside.copy()
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        distance = np.full(shape, np.inf)
        for inverse, boundary in zip(cut.inverse_lengths, cut.boundaries):
            if boundary is not None:
                i, j, ei, ej, x, y, sign = boundary
                np.minimum.at(distance, (ei, ej), 1 / inverse[i, j])
        candidates = {}
        for axis, (inverse, boundary) in enumerate(zip(cut.inverse_lengths, cut.boundaries)):
            if boundary is None:
                continue
            i, j, ei, ej, x, y, sign = boundary
            for k in range(len(i)):
                a, b = int(ei[k]), int(ej[k])
                length = 1 / inverse[i[k], j[k]]
                nodes = grid.x if axis == 0 else grid.y
                edge = i[k] if axis == 0 else j[k]
                if length / (nodes[edge + 1] - nodes[edge]) >= fraction_threshold:
                    continue
                # Move away from the PEC, along the same coordinate line.
                p, q = a + int(sign[k]) if axis == 0 else a, b + int(sign[k]) if axis == 1 else b
                if not (0 <= p < shape[0] and 0 <= q < shape[1] and free[p, q]):
                    continue
                if distance[p, q] <= distance[a, b]:
                    continue  # Avoid cycles and aggregation across narrow gaps.
                neighbor_length = abs(
                    nodes[(p if axis == 0 else q)] - nodes[(a if axis == 0 else b)]
                )
                local_weight = length / (length + neighbor_length)
                child = int(indices[a, b])
                if child not in candidates or length < candidates[child][0]:
                    candidates[child] = length, int(indices[p, q]), local_weight
        parent = np.arange(size)
        weight = np.ones(size)
        for child, (_, target, coefficient) in candidates.items():
            parent[child], weight[child] = target, coefficient
        # Collapse acyclic chains so each constrained node has one master.
        for child in candidates:
            root, coefficient = child, 1.0
            while parent[root] != root:
                coefficient *= weight[root]
                root = parent[root]
            parent[child], weight[child] = root, coefficient
        free_flat = free.ravel()
        self.masters = np.flatnonzero(free_flat & (parent == np.arange(size)))
        self.slaves = np.flatnonzero(free_flat & (parent != np.arange(size)))
        self.roots = parent[self.slaves]
        self.slave_weights = weight[self.slaves]
        compact = np.full(size, -1, dtype=int)
        compact[self.masters] = np.arange(len(self.masters))
        owners = np.full(size, -1, dtype=int)
        owners[free_flat] = compact[parent[free_flat]]
        self.slave_owners = owners[self.slaves]
        dual = [np.diff(np.r_[a[0], (a[:-1] + a[1:]) / 2, a[-1]]) for a in (grid.x, grid.y)]
        mass = np.outer(*dual).ravel()
        self.master_mass = mass[self.masters]
        self.slave_mass_weight = mass[self.slaves] * self.slave_weights
        self.mass = self.master_mass + np.bincount(
            self.slave_owners,
            weights=self.slave_mass_weight * self.slave_weights,
            minlength=len(self.masters),
        )

        # Assemble the projected vacuum stiffness only for a rigorous CFL bound.
        # It is not used for time stepping, which retains sparse local transfers.
        rows, cols, values = [], [], []
        for axis, inverse in enumerate(cut.inverse_lengths):
            left = indices[:-1, :] if axis == 0 else indices[:, :-1]
            right = indices[1:, :] if axis == 0 else indices[:, 1:]
            coefficient = inverse * (dual[1][None, :] if axis == 0 else dual[0][:, None])
            left, right, coefficient = left.ravel(), right.ravel(), coefficient.ravel()
            op, oq = owners[left], owners[right]
            wp, wq = weight[left], weight[right]
            for own, w in ((op, wp), (oq, wq)):
                valid = own >= 0
                rows.extend(own[valid])
                cols.extend(own[valid])
                values.extend((coefficient * w * w)[valid])
            valid = (op >= 0) & (oq >= 0)
            for row, col in ((op, oq), (oq, op)):
                rows.extend(row[valid])
                cols.extend(col[valid])
                values.extend((-coefficient * wp * wq)[valid])
        self.stiffness = coo_matrix(
            (values, (rows, cols)), shape=(len(self.masters), len(self.masters))
        ).tocsr()
        self.stiffness.eliminate_zeros()
        bound = float(np.max(np.asarray(abs(self.stiffness).sum(axis=1)).ravel() / self.mass))
        self.stable_dt = 2 / (C0 * np.sqrt(bound))
        all_x, all_y = np.broadcast_arrays(grid.x[:, None], grid.y[None, :])
        self.slave_positions = all_x.ravel()[self.slaves], all_y.ravel()[self.slaves]
        self.root_positions = all_x.ravel()[self.roots], all_y.ravel()[self.roots]

    def initialize(self, ez, source):
        self.slave_delay = source.retardation(*self.slave_positions)
        self.root_delay = source.retardation(*self.root_positions)
        self.previous_g = self._incident_offset(source, 0.0)
        ez.ravel()[self.slaves] = self.slave_weights * ez.ravel()[self.roots] + self.previous_g

    def _incident_offset(self, source, time):
        return self.slave_weights * source.pulse(time - self.root_delay) - source.pulse(
            time - self.slave_delay
        )

    def advance(self, ez, unconstrained_increment, source, time):
        g = self._incident_offset(source, time)
        delta = unconstrained_increment.ravel()
        numerator = self.master_mass * delta[self.masters] + np.bincount(
            self.slave_owners,
            weights=self.slave_mass_weight * (delta[self.slaves] - (g - self.previous_g)),
            minlength=len(self.masters),
        )
        flat = ez.ravel()
        flat[self.masters] += numerator / self.mass
        flat[self.slaves] = self.slave_weights * flat[self.roots] + g
        self.previous_g = g
