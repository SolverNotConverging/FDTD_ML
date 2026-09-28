"""Reproducible laptop qualification, never a training-label generator."""

import argparse
import json
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.cases import canonical_case
from fdtdmesh.constants import C0
from fdtdmesh.mesh import AxisCollar, Mesh
from fdtdmesh.scattering import cylinder_amplitude, pattern_error
from fdtdmesh.simulation import Convergence, prepare, run_scattering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qualification"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, patterns, cache = [], {}, {}
    policy = Convergence(check_interval=256)

    def run(name, kind="cylinder", ppw=32, graded=False, **kwargs):
        start = perf_counter()
        case, mesh = canonical_case(kind, ppw=ppw, nonuniform=graded, convergence=policy, **kwargs)
        meshing = perf_counter() - start
        result = run_scattering(case, mesh)
        row = dict(
            name=name,
            kind=kind,
            ppw=ppw,
            graded=graded,
            meshing_seconds=meshing,
            **result.diagnostics,
        )
        if kind == "cylinder":
            radius = kwargs.get("radius", 0.47) * C0 / case.frequency
            exact = cylinder_amplitude(radius, case.frequency, case.angles)
            shift = kwargs.get("shift", (0, 0))
            exact *= np.exp(
                2j * np.pi * ((np.cos(case.angles) - 1) * shift[0] + np.sin(case.angles) * shift[1])
            )
            width = 4 / (2 * np.pi * case.frequency / C0) * abs(exact) ** 2
            row.update(
                width_error=pattern_error(result.width[0], width),
                amplitude_error=pattern_error(result.amplitude[0], exact),
            )
        rows.append(row)
        patterns[name] = result.width[0]
        cache[name] = (case, mesh, result)
        result.save(args.output / f"{name}.npz")
        print(
            json.dumps(
                {k: row[k] for k in ("name", "Nt", "gpu_ms", "enlarged_pairs")}
                | {k: row[k] for k in ("width_error", "amplitude_error") if k in row}
            ),
            flush=True,
        )
        return result

    for graded in (False, True):
        for ppw in (16, 32, 64):
            run(f"cylinder_{'graded' if graded else 'uniform'}_{ppw}", ppw=ppw, graded=graded)
    empty = run("empty_graded", "empty", graded=True)
    for kind in ("rectangle", "pair", "slot"):
        for ppw in (16, 32, 64):
            run(f"{kind}_{ppw}", kind, ppw=ppw)
    shifted = run("cylinder_translated", ppw=64, shift=(0.013, 0.019))
    run("cylinder_radius", ppw=64, radius=0.433)
    moved = run("cylinder_contour", ppw=64, contour_offset=1.125)
    case, mesh, baseline = cache["cylinder_uniform_64"]
    # Refine only the PML; preserve every interior line and compare at identical dt.
    collar = AxisCollar(40, case.pml.x.thickness)

    def refine_collar(lines):
        n = len(lines) - 1 + 16
        fixed = collar.fixed_lines(lines[-1], n)
        axis = np.r_[np.zeros(40), lines[32:-32], np.zeros(40)]
        for i, value in fixed.items():
            axis[i] = value
        return axis

    pml_mesh = Mesh(refine_collar(mesh.x), refine_collar(mesh.y))
    pml_case = replace(case, pml=replace(case.pml, x=collar, y=collar))
    pml_dt = prepare(pml_case, pml_mesh)[0].dt
    pml = run_scattering(pml_case, pml_mesh)
    pml_control = run_scattering(case, mesh, dt=pml_dt)
    pml.save(args.output / "cylinder_pml.npz")
    strict = run_scattering(
        replace(case, convergence=replace(policy, stable_checks=6, rtol=1e-8, field_tol=1e-7)), mesh
    )
    single = run_scattering(case, mesh, dtype="float32")
    debug = run_scattering(case, mesh, diagnostic_download=True)
    origin = prepare(case, mesh)[-1]["origin_index"]
    active = (mesh.x >= case.tfsf_box[0]) & (mesh.x <= case.tfsf_box[1])
    incident = debug.debug["incident"][0]
    expected = np.exp(-2j * np.pi * case.frequency / C0 * (mesh.x[active] - case.origin[0]))
    sensitivity = dict(
        contour=pattern_error(moved.width, baseline.width),
        pml_resolution=pattern_error(pml.width, pml_control.width),
        time_step=pattern_error(pml_control.width, baseline.width),
        stopping_tolerance=pattern_error(strict.width, baseline.width),
        float32=pattern_error(single.width, baseline.width),
        incident_complex=pattern_error(incident[active] / incident[origin], expected),
        empty_amplitude_max=float(abs(empty.amplitude).max()),
        translation_width=pattern_error(shifted.width, baseline.width),
    )
    refinements = {}
    for kind in ("rectangle", "pair", "slot"):
        refinements[kind] = [
            pattern_error(patterns[f"{kind}_{a}"], patterns[f"{kind}_{b}"])
            for a, b in ((16, 32), (32, 64))
        ]
    gates = dict(
        cylinder_uniform=next(r["width_error"] for r in rows if r["name"] == "cylinder_uniform_64")
        < 0.002,
        cylinder_graded=next(r["width_error"] for r in rows if r["name"] == "cylinder_graded_64")
        < 0.002,
        empty=sensitivity["empty_amplitude_max"] < 1e-12,
        contour=sensitivity["contour"] < 0.005,
        pml=sensitivity["pml_resolution"] < 0.002,
        time=sensitivity["stopping_tolerance"] < 1e-5,
        timestep=sensitivity["time_step"] < 0.002,
        precision=sensitivity["float32"] < 1e-4,
        incident=sensitivity["incident_complex"] < 0.01,
        translated=next(r["width_error"] for r in rows if r["name"] == "cylinder_translated")
        < 0.003,
        changed_radius=next(r["width_error"] for r in rows if r["name"] == "cylinder_radius")
        < 0.003,
        complex_shapes=all(b < a and b < 0.03 for a, b in refinements.values()),
        gpu_residency=all(
            r["debug_d2h_bytes"] == 0 and r["stepping_field_transfers"] == 0 for r in rows
        ),
    )
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        device=device,
        method=baseline.diagnostics["method"],
        rows=rows,
        sensitivity=sensitivity,
        shape_refinement=refinements,
        gates=gates,
        passed=all(gates.values()),
        controls=dict(
            pml=pml.diagnostics, same_dt=pml_control.diagnostics, extended=strict.diagnostics
        ),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    axes = (fig.add_subplot(121), fig.add_subplot(122, projection="polar"))
    for graded in (False, True):
        selected = [
            r
            for r in rows
            if r["name"].startswith(f"cylinder_{'graded' if graded else 'uniform'}_")
        ]
        axes[0].loglog(
            [r["ppw"] for r in selected],
            [r["width_error"] for r in selected],
            "o-",
            label="Graded" if graded else "Uniform",
        )
    axes[0].set(
        xlabel="Cells per wavelength (nominal budget)",
        ylabel="Relative L2 scattering-width error",
        title="PEC cylinder: spatial convergence",
    )
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()
    exact = cylinder_amplitude(0.47 * C0 / case.frequency, case.frequency, case.angles)
    axes[1].plot(
        case.angles, 4 / (2 * np.pi) * abs(exact) ** 2, "k--", label="Analytical"
    )
    axes[1].plot(
        case.angles, baseline.width[0] / (C0 / case.frequency), label="CUDA, 64 cells/λ"
    )
    axes[1].set(
        ylabel="Scattering width / wavelength",
        title="Absolute bistatic pattern",
    )
    axes[1].legend()
    fig.savefig(args.output / "validation.png", dpi=160)
    print(
        json.dumps(
            dict(sensitivity=sensitivity, shape_refinement=refinements, gates=gates), indent=2
        )
    )
    if not report["passed"]:
        raise SystemExit("Qualification failed; inspect report.json")


if __name__ == "__main__":
    main()
