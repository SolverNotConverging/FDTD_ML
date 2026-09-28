"""Small hybrid accuracy screen, not a qualified reference or training campaign."""

import json
from pathlib import Path

from fdtdmesh import BoundaryPolicy, Geometry, Simulation, SolverSettings
from fdtdmesh.optimization.common import errors


def main():
    root = Path("artifacts/hybrid_validation")
    root.mkdir(parents=True, exist_ok=True)
    geometry = Geometry().added("rectangle", (-0.1, 0.1, -0.1, 0.1))[0]
    geometry = geometry.added("rectangle", (0.0215, 0.0245, 0.07, 0.2), "air")[0]
    rows = []
    results = []
    # Gate fixed before running: hybrid complex-field difference from the fine
    # strict run <10%; no monotone convergence claim for mesh-dependent fallback.
    for label, mode, cells, spacing in (
        ("hybrid100", "hybrid", (100, 100), None),
        ("hybrid120", "hybrid", (120, 120), None),
        ("strict_fine", "conformal", None, 0.299792458 / 128),
    ):
        sim = Simulation(
            fmin=0.9e9,
            fmax=1.1e9,
            boundary=BoundaryPolicy(mode=mode),
            solver=SolverSettings(dft_bins=3),
        )
        sim.set_geometry(geometry)
        if cells:
            sim.apply_mesh("quasi_uniform", cells=cells)
        else:
            sim.apply_mesh("geometry_aware", target_spacing=spacing)
        result = sim.solve()
        result.save(root / f"{label}.h5")
        results.append(result)
        row = dict(
            name=label,
            cells=[result.mesh.Nx, result.mesh.Ny],
            converged=result.converged,
            boundary=result.diagnostics["boundary"],
            steps=result.diagnostics["Nt"],
            gpu_ms=result.diagnostics["gpu_ms"],
        )
        rows.append(row)
        print(label, row["cells"], row["boundary"]["fallback_cells"], flush=True)
    for row, result in zip(rows[:-1], results[:-1]):
        row["difference_from_strict_fine"] = errors(result, results[-1])
        row["passed_screen"] = row["difference_from_strict_fine"]["error"] < 0.1
    output = dict(
        kind="bounded accuracy screen; fine run is not a qualified reference",
        threshold=0.1,
        rows=rows,
    )
    (root / "summary.json").write_text(json.dumps(output, indent=2))
    print([(r["name"], r.get("difference_from_strict_fine", {}).get("error")) for r in rows])
    if not all(r["passed_screen"] for r in rows[:-1]):
        raise SystemExit("Hybrid screen exceeded its predeclared 10% difference gate")


if __name__ == "__main__":
    main()
