"""Independent engineered-shape reference via nested refinement of a valid strict grid.

Unlike the corner-anchor uniform-target projector, this keeps all validated
geometry-aware lines and divides every interval. It is a study protocol, not a
general meshing API.
"""

import json
import sys
from dataclasses import asdict, replace

import numpy as np
from study_cases import apply_seed_mesh, case_directory, make_simulation

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
from fdtdmesh.solver.conformal import UnresolvedGeometryError

angle = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
shape = sys.argv[3] if len(sys.argv) > 3 else "radio_telescope"
root = case_directory(shape, angle, scale)
sim = make_simulation(shape, angle, scale)
base = apply_seed_mesh(sim, shape, angle, scale)
settings = dict(
    method="nested_geometry_aware_subdivision",
    factors=(2, 3, 4, 5, 6, 7),
    rtol=0.005,
    worst_rtol=0.005,
    check_boundaries=True,
)
description = dict(
    schema=1, kind="reference", simulation=simulation_description(sim), settings=settings
)
directory = experiment_directory(root / "subdivision_reference", description)
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


def subdivided(factor):
    def subdivide(v):
        return np.r_[
            np.concatenate(
                [a + (b - a) * np.arange(factor) / factor for a, b in zip(v[:-1], v[1:])]
            ),
            v[-1],
        ]

    x, y = subdivide(base.x), subdivide(base.y)
    p = asdict(sim.pml)
    p.update(
        x=AxisCollar(factor * sim.pml.x.cells, sim.pml.x.thickness),
        y=AxisCollar(factor * sim.pml.y.cells, sim.pml.y.thickness),
    )
    for lines, collar in ((x, p["x"]), (y, p["y"])):
        for i, v in collar.fixed_lines(lines[-1], len(lines) - 1).items():
            lines[i] = v
    trial = clone(
        sim,
        pml=PML(**p),
        layout=replace(sim.layout, exterior_cells=None, scatterer_margin_cells=None),
    )
    trial.apply_mesh(Mesh(x, y))
    return trial


def accepted(metric):
    return (
        metric["error"] <= settings["rtol"] and metric["worst_frequency"] <= settings["worst_rtol"]
    )


last = finest = fine_sim = None
passing = 0
for factor in settings["factors"]:
    try:
        fine_sim = subdivided(factor)
        finest, path, cached = run_cached(fine_sim, root / "subdivision_runs")
    except (UnresolvedGeometryError, MeshInfeasibleError, MeshOptimizationError) as exc:
        report["levels"].append(dict(factor=factor, status=type(exc).__name__, message=str(exc)))
        save()
        print("LEVEL", factor, type(exc).__name__, flush=True)
        continue
    difference = errors(finest, last) if last is not None else None
    passing = passing + 1 if difference is not None and accepted(difference) else 0
    report["levels"].append(
        dict(
            factor=factor,
            cells=(fine_sim.mesh.Nx, fine_sim.mesh.Ny),
            status="converged",
            difference=difference,
            cached=cached,
            dt=finest.diagnostics["dt"],
        )
    )
    report["result_file"] = "result.h5"
    last = finest
    save()
    print(
        "LEVEL",
        factor,
        report["levels"][-1]["cells"],
        None if difference is None else difference["error"],
        flush=True,
    )
    if passing >= 2:
        break

if passing < 2:
    report["status"] = "spatial_unqualified"
    save()
    raise SystemExit("Spatial convergence did not qualify")

stop = fine_sim.settings.stop
tighter = clone(
    fine_sim,
    stop=replace(stop, rtol=stop.rtol * 0.1, atol=stop.atol * 0.1, field_tol=stop.field_tol * 0.1),
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
    comparison, _, cached = run_cached(check, root / "subdivision_runs")
    metric = errors(comparison, finest)
    report["checks"][name] = dict(**metric, passed=accepted(metric), cached=cached)
    save()
    print("CHECK", name, metric["error"], metric["worst_frequency"], flush=True)

report["observed_reference_difference"] = max(
    [x["difference"]["error"] for x in report["levels"] if x.get("difference")][-2:]
    + [x["error"] for x in report["checks"].values()]
)
report["qualified"] = all(x["passed"] for x in report["checks"].values())
report["status"] = "qualified" if report["qualified"] else "sensitivity_unqualified"
finest.save(directory / "result.h5")
save()
(root / "baseline.json").write_text(
    json.dumps(
        dict(
            shape=shape,
            angle=angle,
            scale=scale,
            cells=(base.Nx, base.Ny),
            anchors=base.metadata["geometry_aware"]["witness_anchors"],
            reference_qualified=report["qualified"],
            reference_directory=str(directory),
        )
    ),
    encoding="utf8",
)
print("REFERENCE", report["status"], report["observed_reference_difference"], directory, flush=True)
