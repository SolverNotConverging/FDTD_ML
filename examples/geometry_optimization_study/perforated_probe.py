"""Additional exact PEC/air complexity family for geometry-aware screening."""

import json
from pathlib import Path
from time import perf_counter

from fdtdmesh import Simulation
from fdtdmesh.geometry_mesher import GeometryMeshingError
from fdtdmesh.mesh import MeshInfeasibleError, MeshOptimizationError
from fdtdmesh.solver.conformal import UnresolvedGeometryError

out = Path("artifacts/geometry_optimization_study/perforated_feasibility.jsonl")
for n in (3, 5, 7, 9):
    for angle in (0, 22.5):
        sim = Simulation(fmin=0.9e9, fmax=1.1e9)
        spacing = 0.065
        radius = 0.018
        width = (n - 1) * spacing + 2 * (radius + 0.02)
        sim.add_rectangle((-width / 2, width / 2), (-width / 2, width / 2))
        for i in range(n):
            for j in range(n):
                sim.add_circle(
                    ((i - (n - 1) / 2) * spacing, (j - (n - 1) / 2) * spacing),
                    radius,
                    material="air",
                )
        if angle:
            sim.set_geometry(sim.geometry.rotated(-angle * 3.141592653589793 / 180, origin=(0, 0)))
        root = Path("artifacts/geometry_optimization_study") / f"perforated_{n}_{angle:g}"
        root.mkdir(exist_ok=True)
        sim.geometry.save(root / "geometry.json")
        start = perf_counter()
        rec = dict(n=n, angle=angle, primitive_count=n * n + 1, width=width)
        try:
            mesh = sim.apply_mesh("geometry_aware", time_limit=45, max_cells=(512, 512))
            rec.update(
                status="valid",
                cells=[mesh.Nx, mesh.Ny],
                passes=len(mesh.metadata["geometry_aware"]["passes"]),
                witness_anchors=[
                    len(a) for a in mesh.metadata["geometry_aware"]["witness_anchors"]
                ],
            )
        except (
            GeometryMeshingError,
            MeshInfeasibleError,
            MeshOptimizationError,
            UnresolvedGeometryError,
            ValueError,
        ) as exc:
            rec.update(status=type(exc).__name__, message=str(exc))
        rec["seconds"] = round(perf_counter() - start, 2)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
