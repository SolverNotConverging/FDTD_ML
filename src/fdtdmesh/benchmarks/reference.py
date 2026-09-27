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
    clone,
    errors,
    experiment_key,
    identity,
    read_json,
    refined,
    run_cached,
    write_json,
)


@dataclass(frozen=True)
class ReferenceSettings:
    ppw: tuple = (48, 72, 108, 164)
    rtol: float = 0.002
    worst_rtol: float = 0.005
    max_seconds: float = 900.0
    check_boundaries: bool = True

    def __post_init__(self):
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
        directory = Path(directory)
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
    other.apply_mesh(Mesh(x, y))
    return other


def qualify_reference(sim, *, directory, settings=None, progress=None):
    """Resume cached fine solves; refuse qualification without all requested checks.

    Wall-time limits are checked between solves, never interrupt a native CUDA run.
    An incomplete reference can be inspected but cannot be used by optimize_mesh.
    """
    settings = settings or ReferenceSettings()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = identity(dict(case=experiment_key(sim), settings=asdict(settings)))
    report_path = directory / "reference.json"
    if report_path.exists():
        previous = read_json(report_path)
        if previous["key"] != key:
            raise ValueError("Reference directory belongs to different settings/geometry")
        if previous["qualified"]:
            return Reference.load(directory)
    report = dict(
        key=key,
        case=experiment_key(sim),
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

    for ppw in settings.ppw:
        if perf_counter() - started > settings.max_seconds:
            report["status"] = "time_limit"
            break
        try:
            fine_sim = refined(sim, ppw)
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
            continue
        metric = errors(finest, last) if last is not None else None
        passing = passing + 1 if metric is not None and accepted(metric) else 0
        report["levels"].append(
            dict(
                ppw=ppw,
                status="converged",
                difference=metric,
                cached=cached,
                dt=finest.diagnostics["dt"],
            )
        )
        report["result_file"] = str(path.relative_to(directory))
        last = finest
        save()
        if progress:
            progress(report["levels"][-1])
        if passing >= 2:
            break
    if passing < 2:
        if report["status"] == "running":
            report["status"] = "spatial_unqualified"
        save()
        return Reference(finest, report, directory)
    checks = {}
    try:
        stop = fine_sim.settings.stop
        tighter = clone(
            fine_sim,
            stop=replace(
                stop, rtol=stop.rtol * 0.1, atol=stop.atol * 0.1, field_tol=stop.field_tol * 0.1
            ),
        )
        tighter.apply_mesh(fine_sim.mesh)
        checks["temporal"] = tighter
        if settings.check_boundaries:
            checks["pml"] = _expanded_pml(fine_sim, ppw)
            h = fine_sim.wavelength / ppw
            shift = max(1, round(ppw / 8)) * h
            a, b, c, d = fine_sim.layout.contour_box
            layout = replace(
                fine_sim.layout,
                contour_box=tuple(
                    float(lines[np.argmin(abs(lines - target))])
                    for lines, target in (
                        (fine_sim.mesh.x, a + shift),
                        (fine_sim.mesh.x, b - shift),
                        (fine_sim.mesh.y, c + shift),
                        (fine_sim.mesh.y, d - shift),
                    )
                ),
            )
            contour = clone(fine_sim, layout=layout)
            contour.apply_mesh(fine_sim.mesh)
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
        + [check["error"] for check in report["checks"].values()]
    )
    report["qualified"] = settings.check_boundaries and all(
        c["passed"] for c in report["checks"].values()
    )
    report["status"] = "qualified" if report["qualified"] else "sensitivity_unqualified"
    save()
    return Reference(finest, report, directory)
