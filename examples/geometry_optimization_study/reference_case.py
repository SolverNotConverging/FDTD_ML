"""Qualified GPU references for paired geometry-seeded mesh comparisons."""

import json
import sys

from study_cases import apply_seed_mesh, case_directory, make_simulation

from fdtdmesh.benchmarks import ReferenceSettings, qualify_reference
from fdtdmesh.solver.tmz import cuda_backend

cuda_backend()
shape = sys.argv[1]
angle = float(sys.argv[2])
scale = float(sys.argv[3])
root = case_directory(shape, angle, scale)
sim = make_simulation(shape, angle, scale)
base = apply_seed_mesh(sim, shape, angle, scale)
print("BASE", shape, angle, scale, base.Nx, base.Ny, flush=True)


def progress(item):
    d = item.get("difference") or {}
    print("LEVEL", item["ppw"], item["status"], d.get("error"), flush=True)


settings = ReferenceSettings(
    ppw=(48, 72, 108, 164, 196, 256, 320), rtol=0.005, worst_rtol=0.005, max_seconds=300
)
for attempt in range(1, 4):
    ref = qualify_reference(sim, directory=root / "reference", settings=settings, progress=progress)
    if ref.qualified or ref.report["status"] != "time_limit":
        break
    print("REFERENCE PASS", attempt, "restarting from cached solves", flush=True)
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
