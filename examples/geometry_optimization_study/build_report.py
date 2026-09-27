"""Collect cached study evidence and render reproducible figures.

Run after the screening, reference, and optimization scripts. Outputs are
committed under docs; full HDF5 solver archives stay under ignored artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.benchmarks import Reference
from fdtdmesh.benchmarks.engineered_shapes import ENGINEERED_SHAPES
from fdtdmesh.benchmarks.optimize import Optimization
from fdtdmesh.result import Result

ROOT = Path("artifacts/geometry_optimization_study")
DOCS = Path("docs")
FIGURES = DOCS / "figures" / "geometry_optimization"
FIGURES.mkdir(parents=True, exist_ok=True)


def jsonl(name):
    return [json.loads(s) for s in (ROOT / name).read_text(encoding="utf8").splitlines()]


def key(row):
    return (row["shape"], row["angle"], row["scale"])


catalog = list({key(x): x for x in jsonl("feasibility.jsonl")}.values())
engineered = list({key(x): x for x in jsonl("engineered_feasibility.jsonl")}.values())
perforated = list({(x["n"], x["angle"]): x for x in jsonl("perforated_feasibility.jsonl")}.values())
comparisons = []
for path in sorted(ROOT.glob("*/comparison.json")):
    item = json.loads(path.read_text(encoding="utf8"))
    archive = Optimization.load(item["directory"])
    trials = archive.report["trials"]
    baselines = [
        x
        for x in trials
        if x["kind"] in ("geometry_aware", "uniform", "deterministic") and x["status"] == "feasible"
    ]
    best_baseline = min(baselines, key=lambda x: x["error"])
    feasible_search = [
        x for x in trials if x["kind"] == "differential_evolution" and x["status"] == "feasible"
    ]
    best_trial = min((x for x in trials if x["status"] == "feasible"), key=lambda x: x["error"])
    ref = Reference.load(
        json.loads((path.parent / "baseline.json").read_text())["reference_directory"]
    )
    gain = item["baseline_error"] - item["best_error"]
    floor = item["observed_reference_difference"]
    item.update(
        best_baseline_kind=best_baseline["kind"],
        best_baseline_error=best_baseline["error"],
        search_feasible=len(feasible_search),
        best_kind=best_trial["kind"],
        gain=gain,
        gain_over_reference_difference=gain / floor if floor else None,
        gain_exceeds_reference_difference=gain > floor,
        reference_method=ref.report["settings"].get("method", "uniform_target_with_exact_anchors"),
        reference_levels=ref.report["levels"],
        reference_checks=ref.report["checks"],
        reference_directory=str(ref.directory),
    )
    comparisons.append(item)

order = [
    ("circle", 0, 0.2),
    ("rectangle", 0, 0.2),
    ("star", 30, 0.2),
    ("moon", 30, 0.2),
    ("wifi", 30, 0.2),
    ("sun", 0, 0.2),
    ("u_shape", 30, 0.2),
    ("circle", 0, 0.4),
    ("sun", 0, 0.4),
    ("swept_aircraft", 0, 1),
    ("propeller_aeroplane", 0, 1),
    ("radio_telescope", 0, 1),
    ("radio_telescope", 30, 1),
]
position = {k: i for i, k in enumerate(order)}
comparisons.sort(key=lambda x: position.get(key(x), 999))
output = dict(
    schema=1,
    physical_setup=dict(
        fmin_hz=0.9e9,
        fmax_hz=1.1e9,
        polarization="TMz",
        material="PEC-air",
        backend="native CUDA",
        catalog_scale_units="metres",
        engineered_scale_units="free-space wavelengths at 1 GHz",
    ),
    search=dict(
        strategy="differential_evolution",
        evaluations=60,
        controls=6,
        population=12,
        seed=0,
        initial_mesh="geometry_aware",
        exact_witness_anchors=True,
        fixed_budget=True,
    ),
    catalog_feasibility=catalog,
    engineered_feasibility=engineered,
    perforated_feasibility=perforated,
    comparisons=comparisons,
)
(DOCS / "geometry_aware_optimization_results.json").write_text(
    json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf8"
)


def title(item):
    units = "λ" if item["shape"] in ENGINEERED_SHAPES else "m"
    return f"{item['shape'].replace('_', ' ')} {item['angle']:g}°, scale {item['scale']:g} {units}"


# Red marker is the measured fine-grid discrepancy; bars are actual gains.
labels = [title(x) for x in comparisons]
gains = 100 * np.array([x["gain"] for x in comparisons])
floors = 100 * np.array([x["observed_reference_difference"] for x in comparisons])
y = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(12, max(7, 0.56 * len(labels))))
ax.barh(
    y,
    gains,
    color=["#238b45" if a > b else "#a8c9b4" for a, b in zip(gains, floors)],
    height=0.65,
    label="observed gain",
)
ax.scatter(
    floors,
    y,
    marker="|",
    s=220,
    linewidths=2.2,
    color="#b73534",
    label="measured reference discrepancy",
    zorder=3,
)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("Reduction in complex far-field relative L2 error (percentage points)")
ax.grid(axis="x", alpha=0.2)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURES / "gain_vs_reference_discrepancy.png", dpi=170)
plt.close(fig)


selection = [
    ("star", 30, 0.2),
    ("wifi", 30, 0.2),
    ("sun", 0, 0.4),
    ("swept_aircraft", 0, 1),
    ("propeller_aeroplane", 0, 1),
    ("radio_telescope", 0, 1),
]
chosen = [next(x for x in comparisons if key(x) == target) for target in selection]
fig, axes = plt.subplots(len(chosen), 2, figsize=(17, 28), constrained_layout=True)
for row, item in enumerate(chosen):
    opt = Optimization.load(item["directory"])
    baseline = next(x for x in opt.report["trials"] if x["kind"] == "geometry_aware")
    seed = Result.load(Path(item["directory"]) / baseline["result_file"])
    for col, result in enumerate((seed, opt.best)):
        ax = axes[row, col]
        result.plot_geometry(mesh=True, units="wavelength", ax=ax)
        a, b, c, d = np.array(result.configuration["layout"]["tfsf_box"]) / (
            299792458.0 / ((result.configuration["fmin"] + result.configuration["fmax"]) / 2)
        )
        padding = 0.08 * max(b - a, d - c)
        ax.set_xlim(a - padding, b + padding)
        ax.set_ylim(c - padding, d + padding)
        ax.set_title(
            f"{title(item)} — {'geometry-aware seed' if col == 0 else 'best fixed-budget mesh'}"
        )
        if ax.get_legend():
            ax.get_legend().remove()
fig.savefig(FIGURES / "selected_meshes.png", dpi=180)
plt.close(fig)


engineered_comparisons = [
    x for x in chosen if x["shape"] in ("swept_aircraft", "propeller_aeroplane", "radio_telescope")
]
fig, axes = plt.subplots(
    1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
)
for ax, item in zip(axes, engineered_comparisons):
    opt = Optimization.load(item["directory"])
    baseline = next(x for x in opt.report["trials"] if x["kind"] == "geometry_aware")
    seed = Result.load(Path(item["directory"]) / baseline["result_file"])
    reference = Reference.load(item["reference_directory"]).result
    mid = len(reference.frequencies) // 2
    norm = np.sqrt(np.mean(abs(reference.far_field[mid]) ** 2))
    theta = np.r_[reference.angles, reference.angles[0]]
    for result, label, color in (
        (seed, "seed error", "#b73534"),
        (opt.best, "optimized error", "#166f9b"),
    ):
        values = abs(result.far_field[mid] - reference.far_field[mid]) / norm
        ax.plot(theta, np.r_[values, values[0]], label=label, color=color)
    ax.set_title(title(item) + "\n1 GHz far-field error")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
fig.savefig(FIGURES / "engineered_far_field_error.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Wrote {len(comparisons)} comparisons and 3 figures to {DOCS}")
