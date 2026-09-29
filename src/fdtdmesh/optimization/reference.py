"""Spatial, boundary and temporal qualification of numerical reference fields."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from ..api import ScatteringLayout
from ..geometry import Geometry, Shape
from ..mesh import AxisCollar, MeshInfeasibleError, MeshOptimizationError
from ..pml import PML
from ..result import Result
from ..simulation import ConvergenceError
from ..solver.conformal import UnresolvedGeometryError
from .common import (
    archive_directory,
    clone,
    errors,
    experiment_directory,
    experiment_key,
    identity,
    physical_key,
    read_json,
    refined,
    run_cached,
    simulation_description,
    write_json,
)


@dataclass(frozen=True)
class ReferenceSettings:
    ppw: tuple = (48, 72, 108, 164)
    method: str = "refine"
    factors: tuple = (2, 3, 4, 5)
    rtol: float = 0.002
    worst_rtol: float = 0.005
    max_seconds: float = 900.0
    check_boundaries: bool = True
    boundary_mode: str = "conformal"

    def __post_init__(self):
        if self.boundary_mode not in ("conformal", "hybrid"):
            raise ValueError("boundary_mode must be conformal or hybrid")
        if self.boundary_mode == "hybrid" and self.method != "subdivide":
            raise ValueError("Hybrid references require method='subdivide' to refine every cell")
        if self.method not in ("refine", "subdivide"):
            raise ValueError("Reference method must be refine or subdivide")
        factors = tuple(self.factors)
        if (
            len(factors) < 3
            or any(isinstance(n, bool) or int(n) != n or n < 1 for n in factors)
            or any(a >= b for a, b in zip(factors, factors[1:]))
        ):
            raise ValueError("Use at least three increasing subdivision factors")
        object.__setattr__(self, "factors", factors)
        levels = tuple(self.ppw)
        if (
            len(levels) < 3
            or any(isinstance(n, bool) or int(n) != n or n < 8 for n in levels)
            or any(a >= b for a, b in zip(levels, levels[1:]))
        ):
            raise ValueError("Use at least three increasing integer reference resolutions")
        if (
            not np.isfinite([self.rtol, self.worst_rtol, self.max_seconds]).all()
            or min(self.rtol, self.worst_rtol, self.max_seconds) <= 0
        ):
            raise ValueError("Reference tolerances/time limit must be positive")
        object.__setattr__(self, "ppw", levels)


@dataclass
class Reference:
    result: Result | None
    report: dict
    directory: Path

    @property
    def qualified(self):
        return self.report["qualified"]

    @classmethod
    def load(cls, directory):
        directory = archive_directory(directory, "reference.json")
        report = read_json(directory / "reference.json")
        result = (
            Result.load(directory / report["result_file"]) if report.get("result_file") else None
        )
        return cls(result, report, directory)


def _expanded_pml(sim, ppw):
    h = sim.wavelength / ppw
    cells = max(1, round(ppw / 4))
    pad = cells * h
    shifted = []
    for shape in sim.geometry.shapes:
        p = shape.parameters
        if shape.kind in ("circle", "ellipse"):
            values = (p[0] + pad, p[1] + pad, *p[2:])
        elif shape.kind == "rectangle":
            values = tuple(v + pad for v in p)
        else:
            values = tuple((x + pad, y + pad) for x, y in p)
        shifted.append(Shape(shape.id, shape.kind, shape.material, values, shape.name))
    geometry = Geometry(tuple(v + 2 * pad for v in sim.size), tuple(shifted), sim.geometry.next_id)
    p = asdict(sim.pml)
    p.update(
        x=AxisCollar(sim.pml.x.cells + cells, sim.pml.x.thickness + pad),
        y=AxisCollar(sim.pml.y.cells + cells, sim.pml.y.thickness + pad),
    )
    layout = ScatteringLayout(
        tuple(v + pad for v in sim.layout.tfsf_box),
        tuple(v + pad for v in sim.layout.contour_box),
        sim.layout.source_x + pad,
        tuple(v + pad for v in sim.layout.origin),
    )
    other = clone(sim, geometry=geometry, pml=PML(**p), layout=layout)
    from ..mesh import Mesh

    x = np.r_[
        np.arange(cells) * h, sim.mesh.x + pad, sim.size[0] + pad + np.arange(1, cells + 1) * h
    ]
    y = np.r_[
        np.arange(cells) * h, sim.mesh.y + pad, sim.size[1] + pad + np.arange(1, cells + 1) * h
    ]
    for lines, collar, length in ((x, other.pml.x, other.size[0]), (y, other.pml.y, other.size[1])):
        for i, v in collar.fixed_lines(length, len(lines) - 1).items():
            lines[i] = v
    other._apply_mesh(Mesh(x, y))
    return other


def qualify_reference(sim, *, directory, settings=None, initial_mesh=None, progress=None):
    """Resume cached fine solves; refuse qualification without all requested checks.

    Wall-time limits are checked between solves, never interrupt a native CUDA run.
    An incomplete reference can be inspected but cannot be used by optimize_mesh.
    """
    settings = settings or ReferenceSettings()
    from ..mesh import Mesh

    physical_case = physical_key(sim)
    original_mesh = sim.mesh
    sim = clone(sim)
    sim._boundary = replace(sim._boundary, mode=settings.boundary_mode)
    seed = initial_mesh if initial_mesh is not None else original_mesh
    if settings.method == "subdivide" and seed is None:
        raise ValueError("subdivide reference requires initial_mesh or an applied mesh")

    def subdivided(factor):
        def axis(v):
            return np.r_[
                np.concatenate([np.linspace(a, b, factor + 1)[:-1] for a, b in zip(v[:-1], v[1:])]),
                v[-1],
            ]

        axes = [axis(seed.x), axis(seed.y)]
        p = asdict(sim.pml)
        p.update(
            x=AxisCollar(factor * sim.pml.x.cells, sim.pml.x.thickness),
            y=AxisCollar(factor * sim.pml.y.cells, sim.pml.y.thickness),
        )
        for lines, collar in zip(axes, (p["x"], p["y"])):
            for i, value in collar.fixed_lines(lines[-1], len(lines) - 1).items():
                lines[i] = value
        candidate = clone(
            sim,
            pml=PML(**p),
            layout=replace(sim.layout, exterior_cells=None, scatterer_margin_cells=None),
        )
        candidate._apply_mesh(Mesh(*axes))
        return candidate

    description = dict(
        schema=1,
        kind="reference",
        simulation=simulation_description(sim),
        settings=asdict(settings),
        seed=None if settings.method == "refine" else dict(x=seed.x.tolist(), y=seed.y.tolist()),
        protocol_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    directory = experiment_directory(directory, description)
    key = identity(description)
    report_path = directory / "reference.json"
    if report_path.exists():
        previous = read_json(report_path)
        if previous["key"] != key:
            raise ValueError("Reference archive is inconsistent with its experiment.json")
        if previous["qualified"]:
            return Reference.load(directory)
    report = dict(
        key=key,
        description=description,
        case=experiment_key(sim),
        physical_case=physical_case,
        settings=asdict(settings),
        qualified=False,
        levels=[],
        checks={},
        status="running",
        result_file=None,
    )
    started = perf_counter()
    last = finest = fine_sim = None
    passing = 0

    def save():
        report["elapsed_seconds"] = perf_counter() - started
        write_json(report_path, report)

    def accepted(metric):
        return metric["error"] <= settings.rtol and metric["worst_frequency"] <= settings.worst_rtol

    def localization():
        levels = [level for level in report["levels"] if level["status"] == "converged"]
        if not levels:
            return dict(passed=False)
        end = levels[-1]["boundary"]
        area = end.get("fallback_area_m2", 0.0)
        diameter = end.get("fallback_max_diameter_m", 0.0)
        earlier = [level["boundary"] for level in levels[:-1]]
        passed = area == 0 or any(
            area < b.get("fallback_area_m2", 0.0) * (1 - 1e-8)
            and diameter < b.get("fallback_max_diameter_m", 0.0) * (1 - 1e-8)
            for b in earlier
        )
        return dict(
            passed=passed,
            fallback_area_m2=area,
            fallback_max_diameter_m=diameter,
            note="Physical patches must shrink under subdivision; this is not an error bound",
        )

    for resolution in settings.ppw if settings.method == "refine" else settings.factors:
        ppw = resolution
        if perf_counter() - started > settings.max_seconds:
            report["status"] = "time_limit"
            break
        try:
            fine_sim = refined(sim, ppw) if settings.method == "refine" else subdivided(resolution)
            if settings.method == "subdivide":
                ppw = max(
                    8, round(sim.wavelength / (fine_sim.pml.x.thickness / fine_sim.pml.x.cells))
                )
            finest, path, cached = run_cached(fine_sim, directory / "runs")
        except (
            UnresolvedGeometryError,
            MeshInfeasibleError,
            MeshOptimizationError,
            ConvergenceError,
        ) as exc:
            report["levels"].append(dict(ppw=ppw, status=type(exc).__name__, message=str(exc)))
            passing, last = 0, None
            save()
            if progress:
                progress(report["levels"][-1])
            continue
        metric = errors(finest, last) if last is not None else None
        passing = passing + 1 if metric is not None and accepted(metric) else 0
        report["levels"].append(
            dict(
                ppw=ppw,
                resolution=resolution,
                status="converged",
                difference=metric,
                cached=cached,
                dt=finest.diagnostics["dt"],
                boundary=finest.diagnostics.get("boundary", {}),
            )
        )
        report["result_file"] = str(path.relative_to(directory))
        last = finest
        save()
        if progress:
            progress(report["levels"][-1])
        if passing >= 2 and (settings.boundary_mode == "conformal" or localization()["passed"]):
            break
    if passing < 2:
        if report["status"] == "running":
            report["status"] = "spatial_unqualified" if finest is not None else "no_valid_reference"
        save()
        return Reference(finest, report, directory)
    checks = {}
    if settings.boundary_mode == "hybrid":
        report["checks"]["fallback_localization"] = localization()
    try:
        stop = fine_sim.settings.stop
        tighter = clone(
            fine_sim,
            stop=replace(
                stop,
                rtol=stop.rtol * 0.1,
                atol=stop.atol * 0.1,
                field_tol=stop.field_tol * 0.1,
                stable_checks=stop.stable_checks + 3,
            ),
        )
        tighter._apply_mesh(fine_sim.mesh)
        checks["temporal"] = tighter
        if settings.check_boundaries:
            checks["pml"] = _expanded_pml(fine_sim, ppw)
            h = fine_sim.wavelength / ppw
            shift = max(1, round(ppw / 8)) * h
            a, b, c, d = fine_sim.layout.contour_box

            def moved(lines, target, boundary, side):
                index = int(np.argmin(abs(lines - target)))
                edge = int(np.argmin(abs(lines - boundary)))
                index = min(index, edge - 2) if side < 0 else max(index, edge + 2)
                return float(lines[index])

            layout = replace(
                fine_sim.layout,
                contour_box=tuple(
                    moved(lines, target, boundary, side)
                    for lines, target, boundary, side in (
                        (fine_sim.mesh.x, a + shift, fine_sim.layout.tfsf_box[0], -1),
                        (fine_sim.mesh.x, b - shift, fine_sim.layout.tfsf_box[1], 1),
                        (fine_sim.mesh.y, c + shift, fine_sim.layout.tfsf_box[2], -1),
                        (fine_sim.mesh.y, d - shift, fine_sim.layout.tfsf_box[3], 1),
                    )
                ),
                exterior_cells=None,
                scatterer_margin_cells=None,
            )
            contour = clone(fine_sim, layout=layout)
            contour._apply_mesh(fine_sim.mesh)
            checks["contour"] = contour
        for name, check in checks.items():
            if perf_counter() - started > settings.max_seconds:
                report["status"] = "time_limit"
                save()
                return Reference(finest, report, directory)
            comparison, _, cached = run_cached(check, directory / "runs")
            metric = errors(comparison, finest)
            report["checks"][name] = dict(**metric, passed=accepted(metric), cached=cached)
            save()
    except (
        UnresolvedGeometryError,
        MeshInfeasibleError,
        MeshOptimizationError,
        ConvergenceError,
    ) as exc:
        report["status"] = "check_failed"
        report["failure"] = str(exc)
        save()
        return Reference(finest, report, directory)
    report["observed_reference_difference"] = max(
        [level["difference"]["error"] for level in report["levels"][-2:]]
        + [check["error"] for check in report["checks"].values() if "error" in check]
    )
    report["qualified"] = settings.check_boundaries and all(
        c["passed"] for c in report["checks"].values()
    )
    report["status"] = "qualified" if report["qualified"] else "sensitivity_unqualified"
    save()
    return Reference(finest, report, directory)
