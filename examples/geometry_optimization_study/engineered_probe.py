"""CPU feasibility and cell cost for engineered exact PEC silhouettes."""

import json
from pathlib import Path
from time import perf_counter

from fdtdmesh import Simulation
from fdtdmesh.benchmarks import ENGINEERED_SHAPES, make_engineered_geometry
from fdtdmesh.geometry_mesher import GeometryMeshingError
from fdtdmesh.mesh import MeshInfeasibleError, MeshOptimizationError
from fdtdmesh.solver.conformal import UnresolvedGeometryError

out = Path("artifacts/geometry_optimization_study/engineered_feasibility.jsonl")
for shape in ENGINEERED_SHAPES:
    for angle in (0, 30):
        for scale in (0.5, 1.0):
            sim = Simulation(fmin=0.9e9, fmax=1.1e9)
            sim.set_geometry(make_engineered_geometry(shape, scale=scale, incidence_deg=angle))
            rec = dict(shape=shape, angle=angle, scale=scale, primitives=len(sim.geometry.shapes))
            started = perf_counter()
            try:
                mesh = sim.apply_mesh("geometry_aware", time_limit=45, max_cells=(512, 512))
                rec.update(
                    status="valid",
                    cells=[mesh.Nx, mesh.Ny],
                    passes=len(mesh.metadata["geometry_aware"]["passes"]),
                    anchors=[len(a) for a in mesh.metadata["geometry_aware"]["witness_anchors"]],
                )
            except (
                GeometryMeshingError,
                MeshInfeasibleError,
                MeshOptimizationError,
                UnresolvedGeometryError,
                ValueError,
            ) as exc:
                rec.update(status=type(exc).__name__, message=str(exc))
                if isinstance(exc, GeometryMeshingError):
                    rec["passes"] = len(exc.report["passes"])
            rec["seconds"] = round(perf_counter() - started, 2)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
