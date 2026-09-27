"""Serial, resumable catalog sweeps and CPU-only optimized-mesh galleries."""

from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .optimize import Optimization, optimize_mesh
from .reference import ReferenceSettings, qualify_reference
from .shapes import SHAPES


def run_sweep(
    shapes=SHAPES,
    incidences=(0.0, 30.0, 60.0, 90.0),
    *,
    directory,
    cells=(192, 192),
    reference_settings=None,
    strategy="differential_evolution",
    feature_anchors=False,
    max_evaluations=60,
    max_seconds=600.0,
    progress=None,
):
    """Run cases serially; existing matching archives resume without losing trials.

    Unknown software/runtime failures propagate. A reference that cannot qualify
    produces an explicit unqualified row, not an optimization against bad data.
    """
    from . import make_simulation

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = read_json(directory / "study.json") if (directory / "study.json").exists() else []
    lookup = {(r["shape"], r["incidence_deg"]): r for r in rows}
    for shape in shapes:
        for angle in incidences:
            angle = float(angle)
            sim = make_simulation(shape, incidence_deg=angle)
            case = directory / f"{shape}_{angle:g}"
            if progress:
                progress(dict(shape=shape, incidence_deg=angle, status="reference"))
            ref = qualify_reference(
                sim,
                directory=case / "reference",
                settings=reference_settings or ReferenceSettings(),
            )
            row = dict(
                shape=shape,
                incidence_deg=angle,
                directory=case.name,
                reference_qualified=ref.qualified,
                reference_directory=str(ref.directory.relative_to(directory)),
                status=ref.report["status"],
                best_error=None,
                baseline_error=None,
            )
            if ref.qualified:

                def update(trial):
                    if progress:
                        progress(dict(shape=shape, incidence_deg=angle, **trial))

                opt = optimize_mesh(
                    sim,
                    ref,
                    cells=cells,
                    strategy=strategy,
                    feature_anchors=feature_anchors,
                    directory=case / "optimization",
                    max_evaluations=max_evaluations,
                    max_seconds=max_seconds,
                    progress=update,
                )
                baselines = [
                    t["error"]
                    for t in opt.report["trials"]
                    if t["kind"] == "uniform" and t.get("error") is not None
                ]
                row.update(
                    best_error=opt.report["best_error"],
                    baseline_error=baselines[0] if baselines else None,
                    status=opt.report["status"],
                    validation=opt.report.get("validation"),
                    optimization_directory=str(opt.directory.relative_to(directory)),
                )
            lookup[(shape, angle)] = row
            rows = list(lookup.values())
            write_json(directory / "study.json", rows)
            if progress:
                progress(row)
    return rows


def plot_gallery(directory, *, shapes=None, incidences=None):
    """Plot saved best meshes or explicit qualification/failure labels; no CUDA."""
    import matplotlib.pyplot as plt

    directory = Path(directory)
    rows = read_json(directory / "study.json")
    shapes = list(dict.fromkeys(r["shape"] for r in rows)) if shapes is None else list(shapes)
    incidences = (
        sorted(set(r["incidence_deg"] for r in rows)) if incidences is None else list(incidences)
    )
    if not shapes or not incidences:
        raise ValueError("Gallery needs at least one shape/incidence")
    lookup = {(r["shape"], r["incidence_deg"]): r for r in rows}
    fig, axes = plt.subplots(
        len(shapes),
        len(incidences),
        figsize=(4 * len(incidences), 3.6 * len(shapes)),
        squeeze=False,
        layout="constrained",
    )
    for i, shape in enumerate(shapes):
        for j, angle in enumerate(incidences):
            ax = axes[i, j]
            row = lookup.get((shape, angle))
            if row is None or row["best_error"] is None:
                ax.text(
                    0.5,
                    0.5,
                    "Not run" if row is None else row["status"].replace("_", "\n"),
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            else:
                opt = Optimization.load(
                    directory
                    / row.get(
                        "optimization_directory", str(Path(row["directory"]) / "optimization")
                    )
                )
                opt.best.plot_geometry(mesh=True, units="wavelength", ax=ax)
                if ax.get_legend():
                    ax.get_legend().remove()
                centre = np.array(opt.best.geometry.size) / (
                    2
                    * (
                        299792458.0
                        / ((opt.best.configuration["fmin"] + opt.best.configuration["fmax"]) / 2)
                    )
                )
                ax.set(xlim=(centre[0] - 1, centre[0] + 1), ylim=(centre[1] - 1, centre[1] + 1))
                ax.text(
                    0.02,
                    0.02,
                    f"error {row['best_error']:.2%}",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                )
            ax.set_title(f"{shape} · incidence {angle:g}°")
    return fig
