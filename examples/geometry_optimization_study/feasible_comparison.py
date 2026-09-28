"""Compare feasible local search with archived (or freshly rerun) DE studies.

Run from the repository root. References are qualified under the current source
fingerprint. Historical DE fields are explicitly rescored against that reference;
they are never relabelled as new solver runs or reused as a matching cache entry.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from study_cases import make_simulation
from subdivision_protocol import qualify_subdivided

from fdtdmesh.benchmarks import (
    Optimization,
    Reference,
    ReferenceSettings,
    analyze_mesh_adaptivity,
    optimize_mesh,
    qualify_reference,
)
from fdtdmesh.benchmarks.common import errors
from fdtdmesh.result import Result


def compare(row, args):
    name = f"{row['shape']}_{row['angle']:g}"
    root = args.directory / name
    root.mkdir(parents=True, exist_ok=True)
    previous = Optimization.load(row["directory"])
    initial = next(t for t in previous.report["trials"] if t["kind"] == "geometry_aware")
    seed_result = Result.load(previous.directory / initial["result_file"])
    seed = seed_result.mesh
    sim = make_simulation(row["shape"], row["angle"], row["scale"])
    sim.apply_mesh(seed)
    old_reference = Reference.load(row["reference_directory"])
    if old_reference.report["settings"].get("method") == "nested_geometry_aware_subdivision":
        reference = qualify_subdivided(
            sim,
            seed,
            old_reference.report["settings"],
            root / "reference",
            progress=lambda text: print(name, text, flush=True),
        )
    else:
        settings = ReferenceSettings(**old_reference.report["settings"])
        for _ in range(3):
            reference = qualify_reference(
                sim,
                directory=root / "reference",
                settings=settings,
                progress=lambda item: print(
                    name, "REFERENCE", item.get("ppw"), item.get("status"), flush=True
                ),
            )
            if reference.qualified or reference.report["status"] != "time_limit":
                break
    if not reference.qualified:
        raise RuntimeError(f"{name}: reference unqualified ({reference.report['status']})")

    def progress(t):
        if t["number"] <= 3 or t["number"] % 10 == 0:
            p = t.get("proposal", {})
            print(
                name,
                t["number"],
                t["status"],
                t.get("best_error"),
                p.get("accepted_stage"),
                p.get("movement"),
                flush=True,
            )

    kwargs = dict(
        cells=(seed.Nx, seed.Ny),
        initial_mesh=seed,
        max_evaluations=args.evaluations,
        max_seconds=args.max_seconds,
        controls=6,
        population=12,
        seed=args.seed,
        progress=progress,
    )
    if args.rerun_de:
        previous = optimize_mesh(
            sim, reference, directory=root / "de", strategy="differential_evolution", **kwargs
        )
    local = optimize_mesh(
        sim, reference, directory=root / "local", strategy="feasible_local", **kwargs
    )
    # All historical fields are rescored on the current reference, including
    # their running-best curve. This is analysis, not solver-cache reuse.
    history = []
    best = np.inf
    for t in previous.report["trials"][: args.evaluations]:
        if t["status"] == "feasible":
            result = Result.load(previous.directory / t["result_file"])
            best = min(best, errors(result, reference.result)["error"])
        history.append(None if not np.isfinite(best) else float(best))
    search = [
        t
        for t in previous.report["trials"][: args.evaluations]
        if t["kind"] == "differential_evolution"
    ]
    mobility = analyze_mesh_adaptivity(sim, seed)
    (root / "adaptivity.json").write_text(json.dumps(mobility, indent=2))
    summary = dict(
        shape=row["shape"],
        angle=row["angle"],
        cells=[seed.Nx, seed.Ny],
        evaluations=args.evaluations,
        random_seed=args.seed,
        seed_error=errors(seed_result, reference.result)["error"],
        reference_directory=str(reference.directory),
        reference_change=errors(reference.result, old_reference.result)["error"],
        observed_reference_difference=reference.report["observed_reference_difference"],
        de=dict(
            historical=not args.rerun_de,
            directory=str(previous.directory),
            proposals=len(search),
            solved=sum(t["status"] == "feasible" for t in search),
            best_error=history[-1],
            history=history,
        ),
        local=dict(
            directory=str(local.directory),
            best_error=local.report["best_error"],
            status=local.report["status"],
            wall_seconds=local.report["wall_seconds"],
            validation_passed=local.report["validation"].get("passed", False),
            statistics=local.report["search_statistics"],
        ),
        adaptivity=dict(
            directory=str(root / "adaptivity.json"),
            free_lines=[a["free_lines"] for a in mobility["axes"]],
        ),
    )
    (root / "comparison.json").write_text(json.dumps(summary, indent=2))
    print("RESULT", json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", nargs="+", default=["swept_aircraft", "propeller_aeroplane", "radio_telescope"]
    )
    parser.add_argument("--angles", nargs="+", type=float, default=[0, 90])
    parser.add_argument("--evaluations", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun-de", action="store_true")
    parser.add_argument("--directory", type=Path, default=Path("artifacts/feasible_mesh_study"))
    args = parser.parse_args()
    if not 4 <= args.evaluations <= 200:
        parser.error("Use 4 to 200 evaluations for the paired archived studies")
    rows = json.loads(Path("docs/geometry_aware_optimization_results.json").read_text())[
        "comparisons"
    ]
    selected = [r for r in rows if r["shape"] in args.shapes and r["angle"] in args.angles]
    if len(selected) != len(args.shapes) * len(args.angles):
        parser.error("Every requested shape/angle must have an archived qualified comparison")
    for row in selected:
        compare(row, args)


if __name__ == "__main__":
    main()
