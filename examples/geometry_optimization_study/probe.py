"""CPU geometry-meshing feasibility screen. Results append safely after each case."""

import json
from pathlib import Path
from time import perf_counter

from fdtdmesh import Simulation
from fdtdmesh.benchmarks.shapes import SHAPES, make_geometry
from fdtdmesh.geometry_mesher import GeometryMeshingError
from fdtdmesh.mesh import MeshInfeasibleError, MeshOptimizationError
from fdtdmesh.solver.conformal import UnresolvedGeometryError

out = Path("artifacts/geometry_optimization_study/feasibility.jsonl")
shapes = SHAPES
cases = [(s, a, z) for s in shapes for a in (0, 30) for z in (0.2, 0.4)]
completed = set()
if out.exists():
    for line in out.read_text().splitlines():
        r = json.loads(line)
        completed.add((r["shape"], r["angle"], r["scale"]))
for shape, angle, scale in cases:
    if (shape, angle, scale) in completed:
        continue
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.set_geometry(make_geometry(shape, size=(2.0, 2.0), scale=scale, incidence_deg=angle))
    rec = dict(shape=shape, angle=angle, scale=scale, primitive_count=len(sim.geometry.shapes))
    start = perf_counter()
    try:
        mesh = sim.apply_mesh("geometry_aware", time_limit=20, max_cells=(512, 512))
        meta = mesh.metadata["geometry_aware"]
        rec.update(
            status="valid",
            cells=[mesh.Nx, mesh.Ny],
            passes=len(meta["passes"]),
            donor_pairs=len(sim.discretization.hx.pairs) + len(sim.discretization.hy.pairs),
            dt=sim.discretization.dt,
        )
        try:
            candidate = Simulation(fmin=sim.fmin, fmax=sim.fmax)
            candidate.set_geometry(sim.geometry)
            candidate.apply_mesh("uniform", cells=(mesh.Nx, mesh.Ny))
            rec["same_budget_uniform"] = "valid"
        except (
            MeshInfeasibleError,
            MeshOptimizationError,
            UnresolvedGeometryError,
            ValueError,
        ) as exc:
            rec["same_budget_uniform"] = type(exc).__name__
    except (
        GeometryMeshingError,
        MeshInfeasibleError,
        MeshOptimizationError,
        UnresolvedGeometryError,
        ValueError,
    ) as exc:
        rec.update(status=type(exc).__name__, message=str(exc))
        if isinstance(exc, GeometryMeshingError):
            rec.update(passes=len(exc.report["passes"]), last_issue=exc.report["issues"][:1])
    rec["seconds"] = round(perf_counter() - start, 3)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
