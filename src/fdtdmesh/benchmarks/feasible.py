"""Seed-relative mesh proposals with fixed-index anchors and local restoration.

The linear projector preserves budget, exterior, spacing and grading. Exact
geometry inspection and the enlarged-cell constructor remain the authority on
conformal feasibility. This searches one anchor-index allocation, not every
possible geometry-aware mesh at a given budget.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import diags, eye, hstack, vstack

from ..geometry_mesher import inspect_mesh
from ..mesh import (
    AxisConstraints,
    Mesh,
    MeshInfeasibleError,
    MeshOptimizationError,
    validate_spacing,
)
from ..result import json_text
from ..solver.conformal import UnresolvedGeometryError, build_conformal


@dataclass(frozen=True)
class FeasibleSettings:
    radius: float = 0.75
    min_radius: float = 0.05
    max_radius: float = 2.0
    min_movement: float = 0.03
    repair_passes: int = 3
    backtracks: int = 3
    exploration: float = 0.3

    def __post_init__(self):
        if (
            not np.isfinite(
                [self.radius, self.min_radius, self.max_radius, self.min_movement, self.exploration]
            ).all()
            or not 0 < self.min_movement <= self.min_radius <= self.radius <= self.max_radius
            or not 0 <= self.exploration <= 1
        ):
            raise ValueError("Invalid feasible-search radii, minimum movement or exploration")
        for n in (self.repair_passes, self.backtracks):
            if isinstance(n, bool) or not np.isfinite(n) or int(n) != n or n < 0:
                raise ValueError("Repair passes and backtracks must be nonnegative integers")


def seed_constraints(sim, mesh):
    return AxisConstraints(
        max_spacing=max(
            *(p.thickness / p.cells for p in (sim.pml.x, sim.pml.y)),
            mesh.metadata["geometry_aware"]["target_spacing"],
        )
    )


class _AxisSpace:
    def __init__(self, lines, fixed, constraints, time_limit):
        self.lines = np.array(lines)
        self.length = lines[-1]
        self.q = lines / self.length
        self.fixed = np.array(sorted(fixed), dtype=int)
        self.free = np.setdiff1d(np.arange(len(lines)), self.fixed)
        self.constraints = constraints
        self.time_limit = time_limit
        h = np.diff(lines)
        self.h = np.r_[h[0], (h[:-1] + h[1:]) / 2, h[-1]]
        n = len(h)
        D = diags([-np.ones(n), np.ones(n)], [0, 1], shape=(n, n + 1)).tocsr()
        r = constraints.max_ratio
        self.A = vstack([D, -D, D[1:] - r * D[:-1], D[:-1] - r * D[1:]]).tocsr()
        maximum = self.length if constraints.max_spacing is None else constraints.max_spacing
        self.b = np.r_[
            np.full(n, maximum / self.length),
            np.full(n, -max(constraints.min_spacing / self.length, 1e-9)),
            np.zeros(2 * (n - 1)),
        ]
        self.bounds = np.column_stack([np.zeros(n + 1), np.ones(n + 1)])
        self.bounds[self.fixed] = self.q[self.fixed, None]
        identity = eye(n + 1, format="csr")
        self.project_A = vstack(
            [
                hstack([self.A, self.A * 0]),
                hstack([identity, -identity]),
                hstack([-identity, -identity]),
            ]
        ).tocsr()
        weights = self.length / self.h
        self.cost = np.r_[np.zeros(n + 1), weights / weights.sum()]
        validate_spacing(lines, constraints)

    def solve(self, cost, A, b, bounds):
        result = linprog(
            cost,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
            options=dict(
                time_limit=self.time_limit,
                primal_feasibility_tolerance=1e-10,
                dual_feasibility_tolerance=1e-10,
            ),
        )
        if result.status == 2:
            raise MeshInfeasibleError("Seed-relative projection is infeasible")
        if not result.success:
            raise MeshOptimizationError(f"Seed-relative projection failed: {result.message}")
        return result.x

    def project(self, target, parent, radius, restore=()):
        if np.array_equal(target, parent):
            return parent.copy()
        bounds = self.bounds.copy()
        bounds[:, 0] = np.maximum(bounds[:, 0], (parent - radius * self.h) / self.length)
        bounds[:, 1] = np.minimum(bounds[:, 1], (parent + radius * self.h) / self.length)
        restored = np.array(sorted(restore), dtype=int)
        bounds[restored] = (parent[restored] / self.length)[:, None]
        # Fixed values are copied from the original seed, including exact corners.
        bounds[self.fixed] = self.q[self.fixed, None]
        q = target / self.length
        out = self.solve(
            self.cost,
            self.project_A,
            np.r_[self.b, q, -q],
            np.vstack([bounds, np.tile([0, None], (len(q), 1))]),
        )[: len(q)]
        out *= self.length
        out[restored] = parent[restored]
        out[self.fixed] = self.lines[self.fixed]
        validate_spacing(out, self.constraints)
        return out

    def mobility(self):
        lower, upper = self.lines.copy(), self.lines.copy()
        for i in self.free:
            cost = np.zeros(len(self.lines))
            cost[i] = 1
            lower[i] = self.solve(cost, self.A, self.b, self.bounds)[i] * self.length
            upper[i] = self.solve(-cost, self.A, self.b, self.bounds)[i] * self.length
        width = np.maximum(0, upper - lower) / self.h
        return dict(
            fixed_indices=self.fixed.tolist(),
            free_lines=len(self.free),
            lower=lower.tolist(),
            upper=upper.tolist(),
            width_cells=width.tolist(),
            mobile_lines=int(np.count_nonzero(width > 1e-5)),
        )


class FeasibleSpace:
    def __init__(self, sim, mesh, constraints=None, anchors=None, time_limit=5.0):
        if not np.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("Projection time limit must be positive")
        self.seed = mesh
        self.geometry = sim.computational_geometry
        construction = mesh.metadata.get("geometry_aware", {})
        if (
            construction.get("status") != "valid"
            or construction.get("geometry_sha256")
            != hashlib.sha256(json_text(self.geometry.as_dict()).encode()).hexdigest()
        ):
            raise ValueError("A validated geometry-aware seed for this exact geometry is required")
        self.constraints = constraints or seed_constraints(sim, mesh)
        witnesses = mesh.metadata["geometry_aware"]["witness_anchors"]
        layout = sim.layout
        coordinates = (
            [*layout.tfsf_box[:2], *layout.contour_box[:2], layout.source_x, layout.origin[0]],
            [*layout.tfsf_box[2:], *layout.contour_box[2:]],
        )
        self.axes = []
        for axis, (lines, pml) in enumerate(zip((mesh.x, mesh.y), (sim.pml.x, sim.pml.y))):
            side = (
                pml.cells + sum(layout.exterior_cells or ()) + (layout.scatterer_margin_cells or 0)
            )
            fixed = set(range(side + 1)) | set(range(len(lines) - side - 1, len(lines)))
            for value in [
                *coordinates[axis],
                *witnesses[axis],
                *(anchors[axis] if anchors is not None else []),
            ]:
                i = int(np.argmin(abs(lines - value)))
                if abs(lines[i] - value) > 64 * np.finfo(float).eps * lines[-1]:
                    raise ValueError("Seed must already contain every requested anchor")
                fixed.add(i)
            self.axes.append(_AxisSpace(lines, fixed, self.constraints, time_limit))
        issues, _ = inspect_mesh(self.geometry, mesh)
        if issues:
            raise MeshInfeasibleError("Seed fails exact edge topology or donor inspection")
        build_conformal(self.geometry.to_scene(), mesh)

    def movement(self, mesh, origin):
        values = [
            (a - b)[s.free] / s.h[s.free]
            for a, b, s in zip((mesh.x, mesh.y), (origin.x, origin.y), self.axes)
        ]
        values = np.concatenate(values)
        return dict(
            max_cells=float(np.max(abs(values), initial=0)),
            rms_cells=float(np.sqrt(np.mean(values**2))) if len(values) else 0.0,
            changed_lines=int(np.count_nonzero(abs(values) > 1e-5)),
        )

    def propose(self, parent, rng, radius, controls, settings):
        targets = []
        mode = str(rng.choice(["smooth", "local", "axis"]))
        active_axis = int(rng.integers(2))
        for axis, (s, p) in enumerate(zip(self.axes, (parent.x, parent.y))):
            t = np.linspace(0, 1, len(p))
            displacement = np.interp(t, np.linspace(0, 1, controls), rng.normal(size=controls))
            if mode == "local" and len(s.free):
                centre = rng.choice(s.free)
                width = rng.uniform(4, max(5, len(s.free) / 3))
                displacement *= np.maximum(0, 1 - abs(np.arange(len(p)) - centre) / width)
            if mode == "axis" and axis != active_axis:
                displacement[:] = 0
            displacement[s.fixed] = 0
            displacement /= max(np.max(abs(displacement), initial=0), 1e-12)
            targets.append(p + radius * s.h * displacement)
        info = dict(
            mode=mode, radius_cells=radius, raw_valid=False, checks=[], restored_lines=[0, 0]
        )
        restored = [set(), set()]
        parent_axes = (parent.x, parent.y)
        original = None
        for attempt in range(1 + settings.repair_passes + settings.backtracks):
            stage = (
                "raw"
                if attempt == 0
                else ("restore" if attempt <= settings.repair_passes else "backtrack")
            )
            try:
                if stage == "backtrack":
                    if original is None:
                        break
                    fraction = 0.5 ** (attempt - settings.repair_passes)
                    axes = [p + fraction * (a - p) for p, a in zip(parent_axes, original)]
                else:
                    axes = [
                        s.project(t, p, radius, fixed)
                        for s, t, p, fixed in zip(self.axes, targets, parent_axes, restored)
                    ]
                if attempt == 0:
                    original = axes
                mesh = Mesh(*axes, {**self.seed.metadata, "strategy": "feasible_local"})
                movement = self.movement(mesh, parent)
                if movement["max_cells"] < settings.min_movement:
                    info["checks"].append(dict(stage=stage, status="negligible", **movement))
                    if attempt == 0:
                        break
                    continue
                issues, _ = inspect_mesh(self.geometry, mesh)
                if not issues:
                    build_conformal(self.geometry.to_scene(), mesh)
                    info.update(
                        raw_valid=attempt == 0,
                        accepted_stage=stage,
                        movement=movement,
                        seed_movement=self.movement(mesh, self.seed),
                    )
                    info["checks"].append(dict(stage=stage, status="valid"))
                    return mesh, info
                info["checks"].append(
                    dict(
                        stage=stage,
                        status="geometry_invalid",
                        issues=dict(Counter(i["kind"] for i in issues)),
                    )
                )
                if stage != "backtrack":
                    for issue in issues:
                        axis = issue["axis"]
                        edge = np.searchsorted(axes[axis], issue["edge"][0])
                        transverse = int(np.argmin(abs(axes[1 - axis] - issue["fixed"])))
                        # Restore the cut edge, adjacent donor edges and scanline.
                        restored[axis].update(
                            range(max(0, edge - 1), min(len(axes[axis]), edge + 3))
                        )
                        restored[1 - axis].add(transverse)
                    info["restored_lines"] = list(map(len, restored))
            except (MeshInfeasibleError, MeshOptimizationError, UnresolvedGeometryError) as exc:
                info["checks"].append(
                    dict(stage=stage, status=type(exc).__name__, message=str(exc))
                )
        return None, info


def analyze_mesh_adaptivity(sim, mesh, *, constraints=None, projection_seconds=5.0):
    """LP coordinate ranges conditional on seed anchor indices; no FDTD solves.

    These are upper bounds that omit nonlinear geometry/donor constraints, not
    simultaneous displacements or an estimate of the globally minimum budget.
    Geometry-valid movement is measured separately by feasible_local trials.
    """
    space = FeasibleSpace(sim, mesh, constraints, time_limit=projection_seconds)
    return dict(
        method="fixed-index-linear-relaxation-v1",
        cells=[mesh.Nx, mesh.Ny],
        geometry_constrained=False,
        axes=[s.mobility() for s in space.axes],
    )


def search_statistics(trials):
    """Separate raw proposal validity, restoration, solve success and diversity."""
    rows = [t for t in trials if t["kind"] == "feasible_local"]
    accepted = [t for t in rows if t["status"] == "feasible"]

    def median(section, field):
        values = [t["proposal"][section][field] for t in accepted]
        return float(np.median(values)) if values else None

    return dict(
        proposals=len(rows),
        raw_valid=sum(t["proposal"]["raw_valid"] for t in rows),
        returned_valid=sum("accepted_stage" in t["proposal"] for t in rows),
        solved=len(accepted),
        unique_solved=len({t["mesh_id"] for t in accepted}),
        internal_checks=sum(len(t["proposal"]["checks"]) for t in rows),
        statuses=dict(Counter(t["status"] for t in rows)),
        median_parent_rms_cells=median("movement", "rms_cells"),
        median_seed_rms_cells=median("seed_movement", "rms_cells"),
        median_changed_lines=median("movement", "changed_lines"),
    )
