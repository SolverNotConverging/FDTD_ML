"""Notebook 3 sharp-tip refinement on native CUDA; caches complete run descriptions."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh import C0, BoundaryPolicy, DFTConvergence, Simulation, SolverSettings
from fdtdmesh.optimization.common import errors, run_cached


def subdivide(seed, factor):
    from fdtdmesh import Mesh
    from fdtdmesh.mesh import AxisCollar
    from fdtdmesh.optimization.common import clone
    from fdtdmesh.pml import PML

    axes = [
        np.r_[
            np.concatenate([np.linspace(a, b, factor + 1)[:-1] for a, b in zip(v[:-1], v[1:])]),
            v[-1],
        ]
        for v in (seed.mesh.x, seed.mesh.y)
    ]
    p = asdict(seed.pml)
    p.update(
        x=AxisCollar(factor * seed.pml.x.cells, seed.pml.x.thickness),
        y=AxisCollar(factor * seed.pml.y.cells, seed.pml.y.thickness),
    )
    for axis, collar in zip(axes, (p["x"], p["y"])):
        for index, value in collar.fixed_lines(axis[-1], len(axis) - 1).items():
            axis[index] = value
    sim = clone(
        seed,
        pml=PML(**p),
        layout=replace(seed.layout, exterior_cells=None, scatterer_margin_cells=None),
    )
    sim.apply_mesh("custom", mesh=Mesh(*axes))
    return sim


def make_sim(*, tight=False, exterior_ppw=24):
    from fdtdmesh import DomainPolicy

    lam = C0 / 1e9
    vertices = np.array([(2.25, 2.2), (3.0, 3.0), (2.25, 3.8), (2.45, 3.0)]) * lam
    stop = DFTConvergence(
        stable_checks=6 if tight else 3,
        rtol=1e-6 if tight else 1e-5,
        atol=1e-9 if tight else 1e-8,
        field_tol=1e-6 if tight else 1e-5,
    )
    sim = Simulation(
        fmin=0.95e9,
        fmax=1.05e9,
        solver=SolverSettings(dft_bins=3, stop=stop),
        domain=DomainPolicy(exterior_ppw=exterior_ppw),
        boundary=BoundaryPolicy(mode="hybrid"),
    )
    sim.add_polygon(vertices, name="sharp-tip")
    sim.add_rectangle((3.08 * lam, 3.42 * lam), (2.72 * lam, 3.28 * lam), name="narrow-gap-block")
    return sim


def plots(root, rows, results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    budgets = [r["budget"] for r in rows]
    axes[0].loglog(
        budgets[:-1],
        [r["difference_from_finest"]["error"] for r in rows[:-1]],
        "o-",
        label="Complex field vs finest",
    )
    axes[0].loglog(
        budgets[1:],
        [r["difference_from_previous"]["error"] for r in rows[1:]],
        "s--",
        label="Consecutive complex field",
    )
    axes[0].set(
        xlabel="Total cells per axis",
        ylabel="Relative L2 difference",
        title="Spatial differences (not exact errors)",
    )
    axes[0].legend(fontsize=8)
    axes[1].loglog(budgets, [r["fallback_area_m2"] for r in rows], "o-")
    axes[1].set(
        xlabel="Total cells per axis",
        ylabel="Fallback area [m²]",
        title="Physical size of staircase patches",
    )
    for r, result in zip(rows, results):
        history = result.convergence
        ratio = np.max(
            np.maximum(history.current_error_ratio, history.incident_error_ratio), axis=1
        )
        axes[2].semilogy(history.steps, np.maximum(ratio, 1e-16), label=str(r["budget"]))
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set(
        xlabel="Time step", ylabel="Worst-bin settling ratio", title="Temporal DFT stopping"
    )
    axes[2].legend(title="Cells/axis", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.savefig(root / "convergence.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6), subplot_kw={"projection": "polar"}, layout="constrained"
    )
    for row, result in zip(rows, results):
        centre = len(result.frequencies) // 2
        axes[0].plot(result.angles, result.width[centre] / (C0 / 1e9), label=str(row["budget"]))
        axes[1].plot(result.angles, abs(result.far_field[centre]), label=str(row["budget"]))
    axes[0].set_title("Scattering width / λ₀ at 1 GHz")
    axes[1].set_title("Complex far-field magnitude at 1 GHz")
    for ax in axes:
        ax.legend(title="Cells/axis", loc="upper right", fontsize=8)
    fig.savefig(root / "patterns.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=[120, 180, 260])
    parser.add_argument(
        "--factors",
        nargs="+",
        type=int,
        help="Subdivide the 120-cell seed, including PML; overrides budgets",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/hybrid_tip_refinement"))
    parser.add_argument("--tight", action="store_true")
    parser.add_argument("--exterior-ppw", type=float, default=24)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    rows, results = [], []
    seed = make_sim(tight=args.tight, exterior_ppw=args.exterior_ppw)
    if args.factors:
        seed.apply_mesh("quasi_uniform", cells=(120, 120))
    levels = args.factors or args.budgets
    for level in levels:
        started = perf_counter()
        if args.factors:
            sim = subdivide(seed, level)
        else:
            sim = make_sim(tight=args.tight, exterior_ppw=args.exterior_ppw)
            sim.apply_mesh("quasi_uniform", cells=(level, level))
        budget = sim.mesh.Nx
        boundary = sim.discretization.boundary_report
        print(
            f"START {budget}: {boundary['fallback_cells']} fallback cells; dt={sim.summary()['dt']:.4g}",
            flush=True,
        )
        result, path, cached = run_cached(sim, root / "runs")
        row = dict(
            budget=budget,
            converged=result.converged,
            steps=result.diagnostics["Nt"],
            dt=result.diagnostics["dt"],
            wall_seconds=perf_counter() - started,
            cached=cached,
            result_path=str(path.resolve()),
            fallback_cells=boundary["fallback_cells"],
            fallback_area_m2=boundary["fallback_area_fraction"] * np.prod(sim.size),
            expanded_cells=boundary["expanded_cells"],
            reasons=boundary["reasons"],
            interior_cells=sim.summary().get("scatterer_axis_cells"),
            geometry=seed.geometry.as_dict(),
            domain_allocation=seed.summary()["domain_allocation"],
        )
        if results:
            row["difference_from_previous"] = errors(result, results[-1])
        rows.append(row)
        results.append(result)
        for old, old_result in zip(rows, results):
            old["difference_from_finest"] = errors(old_result, result)
        (root / "summary.json").write_text(
            json.dumps(
                dict(
                    kind="Spatial refinement screen; finest mesh is not qualified ground truth",
                    frequencies=result.frequencies.tolist(),
                    tight=args.tight,
                    exterior_ppw=args.exterior_ppw,
                    refinement="full-domain subdivision"
                    if args.factors
                    else "fixed-exterior budget increase",
                    rows=rows,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"DONE {budget}: steps={row['steps']}, seconds={row['wall_seconds']:.2f}, previous={row.get('difference_from_previous')}",
            flush=True,
        )
        if level == levels[0] or level == levels[-1]:
            from fdtdmesh.plotting import discretization_plot

            config = sim.configuration()
            config["automatic_domain"] = dict(coordinate_offset=seed.coordinate_offset)
            fig = discretization_plot(
                sim.computational_geometry, sim.mesh, sim.discretization, config, "wavelength", None
            )
            fig.axes[0].set(
                xlim=(2.23, 2.31),
                ylim=(3.74, 3.82),
                title=f"Upper sharp PEC tip: {budget} × {budget}, hybrid",
            )
            fig.savefig(root / f"tip_{budget}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
    if len(results) > 1:
        plots(root, rows, results)


if __name__ == "__main__":
    main()
