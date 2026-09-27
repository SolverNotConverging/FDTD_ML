"""Run fixed-budget optimization seeded by the validated geometry-aware grid."""

import json
import sys
from collections import Counter

from study_cases import apply_seed_mesh, case_directory, make_simulation

from fdtdmesh.benchmarks import Reference, optimize_mesh
from fdtdmesh.benchmarks.shapes import SHAPES

shape = sys.argv[1]
angle = float(sys.argv[2])
scale = float(sys.argv[3])
budget = int(sys.argv[4]) if len(sys.argv) > 4 else (60 if shape in SHAPES else 200)
if budget < 3 or budget > 200:
    raise SystemExit("Evaluation budget must be between 3 and 200")
root = case_directory(shape, angle, scale)
sim = make_simulation(shape, angle, scale)
base = apply_seed_mesh(sim, shape, angle, scale)
ref = Reference.load(json.loads((root / "baseline.json").read_text())["reference_directory"])
print("INPUT", shape, angle, scale, base.Nx, base.Ny, "ref", ref.qualified, flush=True)
if not ref.qualified:
    raise SystemExit("Reference is not qualified; optimization skipped")


def progress(item):
    if item["number"] <= 3 or item["number"] % 10 == 0:
        print(
            "TRIAL",
            item["number"],
            item["kind"],
            item["status"],
            item.get("error"),
            item.get("best_error"),
            flush=True,
        )


opt = optimize_mesh(
    sim,
    ref,
    cells=(base.Nx, base.Ny),
    initial_mesh=base,
    directory=root / "optimization",
    max_evaluations=budget,
    max_seconds=1200 if budget > 60 else 240,
    controls=6,
    population=12,
    seed=0,
    progress=progress,
)
summary = dict(
    shape=shape,
    angle=angle,
    scale=scale,
    cells=(base.Nx, base.Ny),
    status=opt.report["status"],
    trials=len(opt.report["trials"]),
    feasible=sum(t["status"] == "feasible" for t in opt.report["trials"]),
    failures=dict(Counter(t["status"] for t in opt.report["trials"] if t["status"] != "feasible")),
    baseline_error=next(
        (t.get("error") for t in opt.report["trials"] if t["kind"] == "geometry_aware"), None
    ),
    best_error=opt.report["best_error"],
    observed_reference_difference=opt.report.get("observed_reference_difference"),
    validation_passed=opt.report.get("validation", {}).get("passed"),
    directory=str(opt.directory),
)
(root / ("comparison.json" if budget == 60 else f"comparison_{budget}.json")).write_text(
    json.dumps(summary, indent=2), encoding="utf8"
)
print("RESULT", json.dumps(summary), flush=True)
