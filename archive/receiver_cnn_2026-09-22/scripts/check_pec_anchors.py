"""PEC-anchor diagnostics: analytical flat-wall cavity and nested pilot grids."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.constants import C0
from fdtdmesh.data.campaign import reference_config
from fdtdmesh.data.schema import SceneSpec, provenance
from fdtdmesh.evaluation.pipeline import grids, metrics, sample_observables, tail_diagnostic
from fdtdmesh.solver.coefficients import build_coefficients


def subdivide(lines):
    out = np.empty(2 * len(lines) - 1)
    out[::2] = lines
    out[1::2] = (lines[:-1] + lines[1:]) / 2
    return out


def controlled(output):
    lx, ly = 0.02, 0.015
    wall = 0.371 * lx
    exact = C0 / 2 * np.sqrt(wall**-2 + ly**-2)
    base = FDTD_2D_Ez(lx, ly, 32, 32, 40e9, Nt=1, dtype="float64")
    base.add_rectangle("PEC", (wall, lx), (0, ly))
    base.add_pec_anchors()
    mesh = base.mesh_from_density([1], [1], time_limit=30)
    x, y = mesh.x, mesh.y
    rows = []
    for n in (32, 64, 128, 256, 512):
        if n > 32:
            x, y = subdivide(x), subdivide(y)
        for anchored in (False, True):
            s = FDTD_2D_Ez(lx, ly, n, n, 40e9, Nt=1, dtype="float64", material_averaging="sampled")
            s.add_rectangle("PEC", (wall, lx), (0, ly))
            if anchored:
                s.add_pec_anchors()
                s.set_mesh(x, y)
            else:
                s.mesh_uniform()
            c = build_coefficients(s, s.mesh, dtype="float64")
            stop = np.flatnonzero(c.pec[1:, n // 2])[0] + 1
            hx = np.diff(s.mesh.x[: stop + 1])
            dual = (hx[:-1] + hx[1:]) / 2
            # y has no interior anchors and remains uniform.
            np.testing.assert_allclose(np.diff(s.mesh.y), ly / n, rtol=1e-10, atol=1e-14)
            ky2 = (2 * np.sin(np.pi / (2 * n)) / (ly / n)) ** 2
            diagonal = (1 / hx[:-1] + 1 / hx[1:]) / dual + ky2
            off = -1 / hx[1:-1] / np.sqrt(dual[:-1] * dual[1:])
            val = eigh_tridiagonal(
                diagonal, off, select="i", select_range=(0, 0), eigvals_only=True
            )[0]
            frequency = np.arcsin(c.dt * C0 * np.sqrt(val) / 2) / (np.pi * c.dt)
            rows.append(
                dict(
                    n=n,
                    anchored=anchored,
                    effective_wall=s.mesh.x[stop],
                    frequency_hz=frequency,
                    analytical_hz=exact,
                    relative_error=abs(frequency / exact - 1),
                )
            )
    (output / "controlled.json").write_text(json.dumps(rows, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for anchored, label in [
        (False, "Uniform: unaligned PEC wall"),
        (True, "Nested mesh: exact PEC-wall anchor"),
    ]:
        rr = [r for r in rows if r["anchored"] == anchored]
        ax.loglog([r["n"] for r in rr], [r["relative_error"] * 100 for r in rr], "o-", label=label)
    ax.set(
        xlabel="Cells per axis",
        ylabel="Frequency error vs analytical [%]",
        title="PEC flat-wall cavity: identical geometry and cell budgets",
    )
    ax.set_xticks([32, 64, 128, 256, 512], labels=["32", "64", "128", "256", "512"])
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output / "controlled.png", dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2), flush=True)


def pilot(args):
    output = args.output
    spec = SceneSpec.from_dict(json.loads(args.manifest.read_text())["scenes"][0])
    saved = np.load(args.base_mesh)
    x, y = saved["x"], saved["y"]
    config = replace(reference_config(), material_averaging="sampled")
    times, frequencies = grids(spec, config)
    previous = None
    rows = []
    (output / "run.json").write_text(
        json.dumps(
            dict(
                scene=spec.to_dict(),
                mode=args.mode,
                base_mesh=str(args.base_mesh.resolve()),
                started_unix=time.time(),
                provenance=provenance(),
            ),
            indent=2,
        )
    )
    for n in args.levels:
        while len(x) - 1 < n:
            x, y = subdivide(x), subdivide(y)
        if len(x) - 1 != n:
            raise ValueError("Levels must refine the saved base by powers of two")
        s = spec.build([n, n], reference=True)
        s.material_averaging = "sampled"
        report = s.add_pec_anchors(mode=args.mode)
        # Canonicalize only floating-point roundoff in fixed PML collar lines.
        # Interior nested lines and physical anchors remain exactly unchanged.
        x, y = x.copy(), y.copy()
        for axis, lines in (("x", x), ("y", y)):
            collar = getattr(s.pml, axis)
            for index, value in collar.fixed_lines(lines[-1], n).items():
                if abs(lines[index] - value) > 1e-12 * lines[-1]:
                    raise ValueError("Refinement changed the physical PML collar")
                lines[index] = value
        s.set_mesh(x, y)
        coeff = build_coefficients(s, s.mesh, dtype="float64")
        nt = int(np.ceil(s.t_end / coeff.dt))
        if n * n * nt > config.max_cell_updates:
            raise RuntimeError("Diagnostic update limit exceeded")
        print("Starting", n, "Nt", nt, flush=True)
        result = s.run()
        observations = sample_observables(result, times)
        row = dict(
            level=n,
            anchors=report,
            diagnostics=result.diagnostics,
            tail=tail_diagnostic(spec, observations, config),
        )
        if previous is not None:
            row["comparison"] = metrics(previous, observations, times, frequencies, config)
        rows.append(row)
        previous = observations
        np.savez_compressed(
            output / f"level_{n}.npz", x=x, y=y, times=times, waveforms=observations
        )
        (output / "levels.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    (output / "complete.json").write_text(json.dumps(dict(finished_unix=time.time())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-mesh", type=Path)
    parser.add_argument("--mode", choices=["axis_aligned", "features"], default="axis_aligned")
    parser.add_argument("--levels", type=int, nargs="+", default=[128, 256, 512, 1024])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.manifest is None:
        controlled(args.output)
    else:
        pilot(args)


if __name__ == "__main__":
    main()
