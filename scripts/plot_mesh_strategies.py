"""Regenerate docs/assets mesh figures from CPU preparation; no FDTD solves."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fdtdmesh import BoundaryPolicy, MeshOptions, Simulation

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets"
COLORS = ("#197a9a", "#aa5937", "#7252a0")


def prepare(strategy):
    sim = Simulation(
        fmin=0.9e9,
        fmax=1.1e9,
        boundary=BoundaryPolicy(mode="conformal" if strategy == "geometry_aware" else "hybrid"),
    )
    sim.add_rectangle((-0.1, 0.1), (-0.1, 0.1))
    sim.add_rectangle((0.0215, 0.0245), (0.07, 0.2), material="air")
    kwargs = {}
    if strategy == "density":
        u = (np.arange(128) + 0.5) / 128
        kwargs["density"] = (
            1 + 5 * np.exp(-(((u - 0.53) / 0.045) ** 2)),
            1 + 3 * np.exp(-(((u - 0.60) / 0.045) ** 2)),
        )
    sim.apply_mesh(strategy, cells=(100, 100), options=MeshOptions(time_limit=30), **kwargs)
    return sim


def save(fig, name, size):
    fig.set_size_inches(*size)
    fig.savefig(OUTPUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.spines.top": False})
    strategies = ("quasi_uniform", "density", "geometry_aware")
    sims = [prepare(strategy) for strategy in strategies]
    summary = []
    for strategy, sim in zip(strategies, sims):
        report = sim.discretization.boundary_report
        summary.append(
            dict(
                strategy=strategy,
                cells=[sim.mesh.Nx, sim.mesh.Ny],
                boundary=report["mode"],
                fallback_cells=report["fallback_cells"],
                geometry_passes=len(sim.mesh.metadata.get("geometry_aware", {}).get("passes", [])),
            )
        )

    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    sim = sims[0]
    sim.plot_geometry(mesh=True, ax=axes[0])
    axes[0].set_title("Automatic domain and fixed exterior axes")
    axes[0].legend(loc="lower left")
    allocation = sim.summary()["domain_allocation"]
    h = allocation["spacing"]
    cells = allocation["cells"]
    actual = allocation["achieved"]
    axes[1].axis("off")
    axes[1].text(
        0,
        0.96,
        "Band: 0.9–1.1 GHz\n"
        f"Centre wavelength: {sim.wavelength * 1000:.2f} mm\n"
        f"Exterior step: {h * 1000:.2f} mm\n\n"
        "From the scatterer outward, per side:\n"
        f"  Scatterer → TFSF: {cells[3]} cells / {actual[3] * 1000:.2f} mm\n"
        f"  TFSF → contour: {cells[2]} cells / {actual[2] * 1000:.2f} mm\n"
        f"  Contour → PML: {cells[1]} cells / {actual[1] * 1000:.2f} mm\n"
        f"  PML thickness: {cells[0]} cells / {actual[0] * 1000:.2f} mm\n\n"
        f"Reserved per axis: {2 * sum(cells)} cells\n"
        f"Inside object bounds: {100 - 2 * sum(cells)} cells/axis\n\n"
        "Green dashed: TFSF\nOrange dashed: NF2FF contour\n"
        "Blue-grey band: PML\nDark fill: exact PEC\n\n"
        "Exterior x/y coordinates are fixed.\nTangential line density can still change\n"
        "because lines span the tensor-product grid.",
        va="top",
        linespacing=1.6,
    )
    save(fig, "mesh_domain.png", (15, 9))

    fig, axes = plt.subplots(2, 3, layout="constrained", gridspec_kw={"height_ratios": [2, 1]})
    for column, (strategy, sim, row, color) in enumerate(zip(strategies, sims, summary, COLORS)):
        ax = axes[0, column]
        sim.plot_geometry(mesh=True, ax=ax)
        ax.get_legend().remove()
        ax.set(xlim=(-0.112, 0.112), ylim=(-0.112, 0.112), xticks=np.linspace(-0.1, 0.1, 5))
        ax.set_title(f"{strategy}\n{row['boundary']}; {row['fallback_cells']} fallback cells")
        axes[1, column].plot(
            (sim.mesh.x[:-1] + sim.mesh.x[1:]) / 2 - sim.coordinate_offset[0],
            np.diff(sim.mesh.x) * 1000,
            color=color,
            lw=1.8,
        )
        axes[1, column].axhline(h * 1000, color="#666666", ls="--", lw=1, label="Exterior step")
        axes[1, column].axvspan(-0.1, 0.1, color=color, alpha=0.08)
        axes[1, column].set(xlabel="x [m]", ylabel="Δx [mm]")
        axes[1, column].grid(alpha=0.2)
        axes[1, column].legend(fontsize=9)
    ymax = 1.08 * max(np.diff(sim.mesh.x).max() * 1000 for sim in sims)
    for ax in axes[1]:
        ax.set_ylim(0, ymax)
    fig.suptitle(
        "Same exact notched PEC and 100 × 100 total budget\n"
        "Grid placement differs; stricter geometry checks require targeted line placement.",
        fontsize=17,
    )
    save(fig, "mesh_strategy_comparison.png", (18, 11))

    fig, axes = plt.subplots(1, 2, layout="constrained")
    for ax, sim, title in zip(
        axes,
        (sims[0], sims[2]),
        (
            "Quasi-uniform + hybrid: staircase patches",
            "Geometry-aware + strict: resolved air notch",
        ),
    ):
        sim.plot_discretization(ax=ax)
        ax.set(xlim=(0.005, 0.040), ylim=(0.06, 0.115), title=title)
        handles, labels = ax.get_legend_handles_labels()
        keep = [(h, label) for h, label in zip(handles, labels) if not label.endswith("_box")]
        handles, labels = map(list, zip(*keep))
        handles.extend(
            [
                Line2D([], [], marker="o", color="#f29d38", linestyle="none"),
                Line2D([], [], color="#d33f49", lw=1.4),
            ]
        )
        labels.extend(["Conformal cut", "Enlarged-cell pair"])
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.37), fontsize=9)
    fig.suptitle(
        "The exact 3 mm air notch stays unchanged in both recipes\n"
        "Red shading marks fallback cells, not an edited geometry.",
        fontsize=16,
    )
    save(fig, "mesh_boundary_detail.png", (13, 10))
    (OUTPUT / "mesh_strategy_figures.json").write_text(
        json.dumps(
            {"kind": "CPU mesh preparation only; no accuracy comparison", "meshes": summary},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
