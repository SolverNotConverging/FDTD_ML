"""CPU-only oblique-incidence proposal checks; no scattering-accuracy claims."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from study_cases import apply_seed_mesh, make_simulation

from fdtdmesh.benchmarks import FeasibleSettings
from fdtdmesh.benchmarks.feasible import FeasibleSpace
from fdtdmesh.strategies import mesh_id

root = Path("artifacts/feasible_mesh_study")
root.mkdir(parents=True, exist_ok=True)
rows = []
for shape, angle in (("swept_aircraft", 45), ("propeller_aeroplane", 15), ("radio_telescope", 30)):
    sim = make_simulation(shape, angle, 1)
    seed = apply_seed_mesh(sim, shape, angle, 1)
    space = FeasibleSpace(sim, seed)
    rng = np.random.default_rng(123)
    details = []
    for _ in range(24):
        mesh, info = space.propose(seed, rng, 0.75, 6, FeasibleSettings())
        if mesh is not None:
            info["mesh_id"] = mesh_id(mesh)
        details.append(info)
    valid = [t for t in details if "mesh_id" in t]
    row = dict(
        shape=shape,
        angle=angle,
        cells=[seed.Nx, seed.Ny],
        trials=24,
        radius_cells=0.75,
        raw_valid=sum(t["raw_valid"] for t in details),
        returned_valid=len(valid),
        unique_valid=len({t["mesh_id"] for t in valid}),
        stages=dict(Counter(t["accepted_stage"] for t in valid)),
        median_rms_cells=float(np.median([t["movement"]["rms_cells"] for t in valid]))
        if valid
        else None,
        details=details,
        fdtd_run=False,
    )
    rows.append(row)
    (root / "oblique_geometry_screen.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({k: v for k, v in row.items() if k != "details"}), flush=True)
