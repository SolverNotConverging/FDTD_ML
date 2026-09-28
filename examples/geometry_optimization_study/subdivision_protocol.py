"""Reusable version of the existing nested engineered-reference study protocol."""

import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from fdtdmesh.benchmarks import Reference
from fdtdmesh.benchmarks.common import (
    clone,
    errors,
    experiment_directory,
    experiment_key,
    identity,
    run_cached,
    simulation_description,
    write_json,
)
from fdtdmesh.benchmarks.reference import _expanded_pml
from fdtdmesh.mesh import AxisCollar, Mesh, MeshInfeasibleError, MeshOptimizationError
from fdtdmesh.pml import PML
from fdtdmesh.simulation import ConvergenceError
from fdtdmesh.solver.conformal import UnresolvedGeometryError


def qualify_subdivided(sim, seed, settings, root, progress=print):
    description = dict(
        schema=1,
        kind="reference",
        simulation=simulation_description(sim),
        settings=settings,
        seed=dict(x=seed.x.tolist(), y=seed.y.tolist()),
        protocol_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    directory = experiment_directory(root, description)
    if (directory / "reference.json").exists():
        existing = Reference.load(directory)
        if existing.qualified:
            return existing
    report = dict(
        key=identity(description),
        description=description,
        case=experiment_key(sim),
        settings=settings,
        qualified=False,
        levels=[],
        checks={},
        status="running",
        result_file=None,
    )

    def save():
        write_json(directory / "reference.json", report)

    def accepted(metric):
        return (
            metric["error"] <= settings["rtol"]
            and metric["worst_frequency"] <= settings["worst_rtol"]
        )

    def subdivided(factor):
        def axis(v):
            return np.r_[
                np.concatenate(
                    [a + (b - a) * np.arange(factor) / factor for a, b in zip(v[:-1], v[1:])]
                ),
                v[-1],
            ]

        x, y = axis(seed.x), axis(seed.y)
        p = asdict(sim.pml)
        p.update(
            x=AxisCollar(factor * sim.pml.x.cells, sim.pml.x.thickness),
            y=AxisCollar(factor * sim.pml.y.cells, sim.pml.y.thickness),
        )
        for lines, collar in ((x, p["x"]), (y, p["y"])):
            for i, v in collar.fixed_lines(lines[-1], len(lines) - 1).items():
                lines[i] = v
        candidate = clone(
            sim,
            pml=PML(**p),
            layout=replace(sim.layout, exterior_cells=None, scatterer_margin_cells=None),
        )
        candidate.apply_mesh(Mesh(x, y))
        return candidate

    last = finest = fine_sim = None
    passing = 0
    for factor in settings["factors"]:
        try:
            candidate = subdivided(factor)
            result, path, cached = run_cached(candidate, directory / "runs")
        except (
            UnresolvedGeometryError,
            MeshInfeasibleError,
            MeshOptimizationError,
            ConvergenceError,
        ) as exc:
            report["levels"].append(
                dict(factor=factor, status=type(exc).__name__, message=str(exc))
            )
            save()
            progress(f"SUBDIVISION {factor} {type(exc).__name__}")
            continue
        fine_sim, finest = candidate, result
        difference = errors(finest, last) if last is not None else None
        passing = passing + 1 if difference is not None and accepted(difference) else 0
        report["levels"].append(
            dict(
                factor=factor,
                cells=[fine_sim.mesh.Nx, fine_sim.mesh.Ny],
                status="converged",
                difference=difference,
                cached=cached,
                dt=finest.diagnostics["dt"],
            )
        )
        report["result_file"] = str(path.relative_to(directory))
        last = finest
        save()
        progress(
            f"SUBDIVISION {factor} difference={None if difference is None else difference['error']}"
        )
        if passing >= 2:
            break
    if passing < 2:
        report["status"] = "spatial_unqualified"
        save()
        return Reference(finest, report, directory)
    stop = fine_sim.settings.stop
    tighter = clone(
        fine_sim,
        stop=replace(
            stop, rtol=stop.rtol * 0.1, atol=stop.atol * 0.1, field_tol=stop.field_tol * 0.1
        ),
    )
    tighter.apply_mesh(fine_sim.mesh)
    checks = {"temporal": tighter}
    ppw = round(sim.wavelength / np.max(np.diff(fine_sim.mesh.x[: fine_sim.pml.x.cells + 1])))
    checks["pml"] = _expanded_pml(fine_sim, ppw)
    shift = sim.wavelength / 8
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
    contour.apply_mesh(fine_sim.mesh)
    checks["contour"] = contour
    for name, check in checks.items():
        comparison, _, cached = run_cached(check, directory / "runs")
        metric = errors(comparison, finest)
        report["checks"][name] = dict(**metric, passed=accepted(metric), cached=cached)
        save()
        progress(f"CHECK {name} difference={metric['error']}")
    report["observed_reference_difference"] = max(
        [x["difference"]["error"] for x in report["levels"] if x.get("difference")][-2:]
        + [x["error"] for x in report["checks"].values()]
    )
    report["qualified"] = settings["check_boundaries"] and all(
        x["passed"] for x in report["checks"].values()
    )
    report["status"] = "qualified" if report["qualified"] else "sensitivity_unqualified"
    save()
    return Reference(finest, report, directory)
