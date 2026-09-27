"""Qualified GPU references for paired geometry-seeded mesh comparisons."""

import json
import sys
from pathlib import Path

from fdtdmesh import Simulation
from fdtdmesh.benchmarks import ReferenceSettings, make_engineered_geometry, qualify_reference
from fdtdmesh.benchmarks.shapes import SHAPES, make_geometry
from fdtdmesh.solver.tmz import cuda_backend

cuda_backend()
shape = sys.argv[1]
angle = float(sys.argv[2])
scale = float(sys.argv[3])
root = Path("artifacts/geometry_optimization_study") / f"{shape}_{angle:g}_{scale:g}"
sim = Simulation(fmin=0.9e9, fmax=1.1e9)
sim.set_geometry(
    make_geometry(shape, size=(2.0, 2.0), scale=scale, incidence_deg=angle)
    if shape in SHAPES
    else make_engineered_geometry(shape, scale=scale, incidence_deg=angle)
)
base = sim.apply_mesh("geometry_aware", time_limit=40, max_cells=(512, 512))
print("BASE", shape, angle, scale, base.Nx, base.Ny, flush=True)


def progress(item):
    d = item.get("difference") or {}
    print("LEVEL", item["ppw"], item["status"], d.get("error"), flush=True)


settings = ReferenceSettings(
    ppw=(48, 72, 108, 164, 196, 256, 320), rtol=0.005, worst_rtol=0.005, max_seconds=300
)
ref = qualify_reference(sim, directory=root / "reference", settings=settings, progress=progress)
print(
    "REFERENCE",
    ref.report["status"],
    ref.qualified,
    ref.report.get("observed_reference_difference"),
    ref.directory,
    flush=True,
)
(root / "baseline.json").write_text(
    json.dumps(
        dict(
            shape=shape,
            angle=angle,
            scale=scale,
            cells=(base.Nx, base.Ny),
            anchors=base.metadata["geometry_aware"]["witness_anchors"],
            reference_qualified=ref.qualified,
            reference_directory=str(ref.directory),
        )
    ),
    encoding="utf8",
)
