"""Publish measured acceptance, diversity and error; never invent missing cases."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.benchmarks import Optimization

ROOT = Path("artifacts/feasible_mesh_study")
FIGURES = Path("docs/figures/feasible_optimization")
FIGURES.mkdir(parents=True, exist_ok=True)
rows = [json.loads(p.read_text()) for p in sorted(ROOT.glob("*/comparison.json"))]
if not rows:
    raise SystemExit("No completed comparisons")

# Also compare prefixes with equal accepted search evaluations. This avoids
# attributing every benefit to the larger number of useful fields after repair.
for row in rows:
    diagnostic_path = ROOT / f"{row['shape']}_{row['angle']:g}" / "adaptivity.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        row["linear_adaptivity"] = []
        for axis in diagnostic["axes"]:
            free = np.ones(len(axis["width_cells"]), dtype=bool)
            free[axis["fixed_indices"]] = False
            widths = np.asarray(axis["width_cells"])[free]
            row["linear_adaptivity"].append(dict(
                free_lines=axis["free_lines"], mobile_lines=axis["mobile_lines"],
                median_range_cells=float(np.median(widths)) if len(widths) else 0.0,
            ))
    de_trials = Optimization.load(row["de"]["directory"]).report["trials"][: row["evaluations"]]
    local_trials = Optimization.load(row["local"]["directory"]).report["trials"]
    row["local"]["median_proposal_seconds"] = float(
        np.median([t["proposal"]["seconds"] for t in local_trials if t["kind"] == "feasible_local"])
    )
    de_indices = [
        i
        for i, t in enumerate(de_trials)
        if t["kind"] == "differential_evolution" and t["status"] == "feasible"
    ]
    local_indices = [
        i
        for i, t in enumerate(local_trials)
        if t["kind"] == "feasible_local" and t["status"] == "feasible"
    ]
    count = min(len(de_indices), len(local_indices))
    if count:
        row["matched_accepted_search_evaluations"] = dict(
            count=count,
            de_error=row["de"]["history"][de_indices[count - 1]],
            local_error=min(
                t["error"]
                for t in local_trials[: local_indices[count - 1] + 1]
                if t.get("error") is not None
            ),
        )

labels = [f"{r['shape'].replace('_', ' ')}\n{r['angle']:g}°" for r in rows]
x = np.arange(len(rows))
fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
for shift, key, title, color in (
    (-0.25, "de", "DE accepted", "#7c8799"),
    (0, "raw", "Local raw valid", "#dd9944"),
    (0.25, "repaired", "Local after repair, solved", "#187b78"),
):
    values = []
    for row in rows:
        s = row["local"]["statistics"]
        values.append(
            100
            * (
                row["de"]["solved"] / row["de"]["proposals"]
                if key == "de"
                else s["raw_valid" if key == "raw" else "solved"] / s["proposals"]
            )
        )
    axes[0].bar(x + shift, values, 0.24, label=title, color=color)
axes[0].set(ylabel="Proposals accepted [%]", ylim=(0, 110), title="Raw and repaired acceptance")
for shift, key, title, color in (
    (-0.25, "seed", "Geometry-aware seed", "#bbb"),
    (0, "de", "DE best", "#7c8799"),
    (0.25, "local", "Local best", "#187b78"),
):
    axes[1].bar(
        x + shift,
        [100 * (r["seed_error"] if key == "seed" else r[key]["best_error"]) for r in rows],
        0.24,
        label=title,
        color=color,
    )
axes[1].set(
    ylabel="Relative complex far-field error [%]", title="Best error against qualified reference"
)
for ax in axes:
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
fig.savefig(FIGURES / "comparison.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(
    len(rows), 2, figsize=(14, 4 * len(rows)), squeeze=False, layout="constrained"
)
details = []
for row, (history_ax, variation_ax) in zip(rows, axes):
    opt = Optimization.load(row["local"]["directory"])
    best = np.inf
    curve = []
    for trial in opt.report["trials"]:
        if trial.get("error") is not None:
            best = min(best, trial["error"])
        curve.append(best * 100)
    history_ax.plot(np.arange(1, len(curve) + 1), curve, color="#187b78", label="Feasible local")
    history_ax.plot(
        np.arange(1, len(row["de"]["history"]) + 1),
        np.array(row["de"]["history"]) * 100,
        color="#7c8799",
        label="DE",
    )
    history_ax.set(
        xlabel="Proposal trial, including baselines",
        ylabel="Best complex error [%]",
        title=f"{row['shape']} {row['angle']:g}°",
    )
    history_ax.legend()
    trials = [
        t
        for t in opt.report["trials"]
        if t["kind"] == "feasible_local" and t["status"] == "feasible"
    ]
    for raw, color, label in (
        (True, "#dd9944", "Raw valid"),
        (False, "#187b78", "Restored/backtracked"),
    ):
        selected = [t for t in trials if t["proposal"]["raw_valid"] == raw]
        variation_ax.scatter(
            [t["proposal"]["seed_movement"]["rms_cells"] for t in selected],
            [100 * t["error"] for t in selected],
            s=14,
            alpha=0.65,
            color=color,
            label=label,
        )
    variation_ax.set(
        xlabel="RMS displacement from seed [seed local cell widths]",
        ylabel="Candidate complex error [%]",
        title="Explored, strictly valid mesh variation",
    )
    variation_ax.legend()
    for ax in (history_ax, variation_ax):
        ax.grid(alpha=0.2)
    de = Optimization.load(row["de"]["directory"])
    # Historical DE winner may differ after reference rescore; report figure
    # uses its archived winner and labels it explicitly.
    fig_mesh, mesh_axes = plt.subplots(1, 2, figsize=(24, 12), layout="constrained")
    for ax, result, label in zip(
        mesh_axes, (de.best, opt.best), ("DE archived winner", "Feasible local winner")
    ):
        result.plot_geometry(mesh=True, units="wavelength", ax=ax)
        bounds = result.geometry.bounds
        lam = 299792458 / np.mean(result.frequencies)
        padx, pady = 0.07 * (bounds[1] - bounds[0]), 0.07 * (bounds[3] - bounds[2])
        ax.set(
            xlim=((bounds[0] - padx) / lam, (bounds[1] + padx) / lam),
            ylim=((bounds[2] - pady) / lam, (bounds[3] + pady) / lam),
            title=f"{label}: {row['shape']} {row['angle']:g}°, {row['cells'][0]} × {row['cells'][1]}",
        )
    filename = f"{row['shape']}_{row['angle']:g}_meshes.png"
    fig_mesh.savefig(FIGURES / filename, dpi=120)
    plt.close(fig_mesh)
    details.append(
        f"![{row['shape']} {row['angle']:g} degree meshes](figures/feasible_optimization/{filename})"
    )
fig.savefig(FIGURES / "history_and_variation.png", dpi=140)
plt.close(fig)

table = [
    "| Shape / incidence | Cells | DE accepted | Local raw valid | Local solved after repair | Seed error | DE best | Local best |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
variation = [
    "| Shape / incidence | Unique local solves | Internal candidate checks | Median parent RMS move | Median seed RMS move | Median moved lines | Search seconds | Winner temporal check |",
    "|---|---:|---:|---:|---:|---:|---:|---|",
]
for r in rows:
    s = r["local"]["statistics"]
    table.append(
        f"| {r['shape']} / {r['angle']:g}° | {r['cells'][0]} × {r['cells'][1]} | {r['de']['solved']}/{r['de']['proposals']} | {s['raw_valid']}/{s['proposals']} | {s['solved']}/{s['proposals']} | {100 * r['seed_error']:.3f}% | {100 * r['de']['best_error']:.3f}% | {100 * r['local']['best_error']:.3f}% |"
    )
    variation.append(
        f"| {r['shape']} / {r['angle']:g}° | {s['unique_solved']} | {s['internal_checks']} | {s['median_parent_rms_cells']:.3f} cells | {s['median_seed_rms_cells']:.3f} cells | {s['median_changed_lines']:.0f} | {r['local']['wall_seconds']:.1f} | {r['local']['validation_passed']} |"
    )
text = """# Feasible local mesh optimization: measured comparison

This experiment tests whether seed-relative proposals and geometric restoration
improve acceptance without collapsing the search into nearly identical meshes.
The physical problem remains exact-geometry, 2D TMz PEC scattering with native
CUDA conformal FDTD, enlarged-cell updates, broadband DFT and GPU NF2FF.
See the [original study](geometry_aware_optimization_study.md) for the numerical
background, exact geometry construction, thin-feature challenges and CNN proposal.

## Algorithm and constraints

`strategy="feasible_local"` starts from an accepted geometry-aware seed. Its
witness coordinates AND their line indices remain fixed, as do exterior/margin
coordinates. Smooth, local and single-axis displacement proposals are projected
onto linear spacing/ratio constraints at exactly the original axis counts.
Proposal radii use the seed's local dual widths, making their scale consistent
as the parent mesh changes. The retained pool combines low-error and displaced
parents; default exploration probability is 0.3.

Every candidate is inspected against exact CSG scanline intersections and the
strict enlarged-cell operator. Failed edge regions can be restored to a valid
parent's coordinates while other changes remain. These restoration constraints
are temporary. Further backtracking tests smaller endpoints, with no assumption
that feasibility is monotone along a segment. A minimum maximum displacement of
0.03 seed-cell widths filters negligible proposals, and exact duplicates are
excluded from local solves. Repairs move grid lines while preserving exact
geometry and enlarged-cell equations. This version keeps anchor indices fixed;
it does not certify a global optimum. Radius adjustment responds to feasibility,
while the retained parent pool uses objective values and geometric diversity.
It is not a formal trust-region stationarity test. Movable interior tensor-grid
lines still extend tangentially through the exterior; the preserved quantities
are the exterior axis coordinates.

## Comparison protocol

Each local search uses the recorded proposal allowance and RNG seed (defaults:
200 trials, including three baselines, and seed 0), six interpolation controls
and a 12-parent pool. It uses
the exact archived geometry-aware seed and cell budget of its corresponding DE
study. Both methods retain the same three comparison baselines; those baselines
can have other anchor-index assignments, while local-search proposals stay in
the seed's allocation. Best-error curves include the baselines. Numerical
references are requalified under the current implementation.
Unless `--rerun-de` was used, DE is a historical comparison: every archived
feasible field is rescored against the current reference, not relabelled as a
new run or reused as a matching solver cache. A new optimizer has more internal
geometry checks and often more GPU solves for the same proposal budget; this
is not an equal-wall-time or equal-solve-budget performance comparison.
Every mesh retains its own native CFL time step. Fixed cell counts therefore
do not imply identical time steps or update counts. The winner check tightens
DFT/field stopping tolerances; it is a truncation-convergence check, not an
independent time-step-refinement study.

Raw acceptance counts proposals valid before restoration. Final acceptance
counts unique proposals that also completed the FDTD convergence criterion.
Internal check counts include projection failures and negligible candidates.
More acceptance alone does not prove a lower achievable field error.

"""
text += (
    "\n".join(table)
    + "\n\n![Acceptance and error](figures/feasible_optimization/comparison.png)\n\n"
)
text += """### Findings and experiment cutoff

The four completed comparisons contain 788 local-search proposals, each producing
a distinct mesh that passed strict geometry/enlarged-cell checks and the FDTD
stopping criterion. Historical DE accepted 584 of 788 proposals (74.1%). The local
method accepted 607 of 788 before restoration (77.0%) and all 788 after restoration
or backtracking. This is measured acceptance for these cases, not a guarantee for
new geometries.

The largest accuracy improvement was the swept aircraft at 0°: relative complex
far-field error decreased from 1.695% to 1.415%. Both propeller cases also improved.
The telescope results were comparable: 1.018% versus 1.024%, a difference smaller
than the observed numerical-reference discrepancy. Median accepted moves ranged
from 0.655 to 0.717 seed local cell widths, with 61–100 moved lines. Thus these
searches retained substantial variation rather than achieving acceptance through
negligible changes. All four winners passed the tighter stopping-tolerance check.

The experiment was stopped at the user's request once this evidence was available.
The partially completed telescope 90° optimization and unstarted aircraft 90°
comparison are excluded from the completed-case tables and conclusions. Cached
partial work remains available for a future explicit continuation. No further
simulation is needed to reproduce this report from the saved results.

"""
text += "### Equal accepted-search prefixes\n\nThese post-hoc prefixes include the same number of accepted search evaluations for each method, with their preceding baselines. Cache hits count as accepted evaluations. They do not equate CPU repair work or GPU runtime.\n\n| Case | Accepted search evaluations | DE best | Local best |\n|---|---:|---:|---:|\n"
for row in rows:
    matched = row.get("matched_accepted_search_evaluations")
    if matched:
        text += f"| {row['shape']} / {row['angle']:g}° | {matched['count']} | {100 * matched['de_error']:.3f}% | {100 * matched['local_error']:.3f}% |\n"
text += "\n## Mesh variation and validation\n\n" + "\n".join(variation)
text += "\n\nRMS displacement is measured over the seed's nonfixed coordinates and normalized by seed local dual cell widths. The seed RMS column measures cumulative departure, while parent RMS measures each accepted move.\n\n"
text += f"The largest current-versus-previous reference field difference is {max(r['reference_change'] for r in rows):.3g} relative L2. Observed reference discrepancies and all archive paths are retained in [the numerical data](feasible_optimizer_results.json). Small differences between optimizer winners must be interpreted relative to those discrepancies.\n\n"
text += (
    "![History and explored variation](figures/feasible_optimization/history_and_variation.png)\n\n## Best meshes\n\n"
    + "\n\n".join(details)
)
text += """

## Adaptivity diagnostics and remaining scope

`analyze_mesh_adaptivity(sim, seed)` returns LP minimum/maximum coordinates under
fixed witness indices, spacing and grading. These are conditional upper bounds:
they omit geometry/donor constraints and their coordinate extremes cannot all be
attained simultaneously. Per-case `adaptivity.json` files store the full arrays.
Neither free-line counts nor acceptance percentages certify the minimum viable
cell budget. A budget sweep and an outer discrete anchor-index allocation search
remain separate future experiments. Solver-time-aware or equal-solve comparisons
and multiple random seeds are also needed before general performance claims.

The same feasibility layer could accept residual mesh proposals from a CNN.
Training targets should retain exact geometry, incidence/frequency conditions,
budget, seed, validation outcome and actual displacement—not only density curves.
This first implementation provides feasible examples; it does not demonstrate a
learned optimizer or establish globally optimal labels.

"""
text += "| Case | Free lines x / y | LP-mobile lines x / y | Median LP range x / y (seed cell widths) |\n|---|---:|---:|---:|\n"
for row in rows:
    if "linear_adaptivity" in row:
        ax, ay = row["linear_adaptivity"]
        text += f"| {row['shape']} / {row['angle']:g}° | {ax['free_lines']} / {ay['free_lines']} | {ax['mobile_lines']} / {ay['mobile_lines']} | {ax['median_range_cells']:.2f} / {ay['median_range_cells']:.2f} |\n"
text += "\nThese LP ranges describe available coordinate movement under the linear constraints only. Actual accepted displacement above provides complementary empirical evidence after exact geometry validation.\n"
text += """

## Reproduction

Run `python examples/geometry_optimization_study/feasible_comparison.py` from the
repository root with the archived engineered study present; `--rerun-de` performs
a new DE search too. Then run
`python examples/geometry_optimization_study/build_feasible_report.py`.
Notebook 04 provides an opt-in API example independent of these archived cases.
See [API argument tables](mesh_optimization_api.md) for all settings.
"""
screen_path = ROOT / "oblique_geometry_screen.json"
if screen_path.exists():
    screen = [
        {k: v for k, v in row.items() if k != "details"}
        for row in json.loads(screen_path.read_text())
    ]
    text += "\n## Additional oblique geometry-only checks\n\nEach case generated 24 proposals from its unchanged valid seed at radius 0.75 local cell widths, RNG seed 123. These check exact topology and enlarged-cell construction only: no FDTD solves, objective optimization or scattering-accuracy claims are included.\n\n| Case | Cells | Raw valid | Valid after restoration | Unique valid | Median RMS movement |\n|---|---|---:|---:|---:|---:|\n"
    for row in screen:
        text += f"| {row['shape']} / {row['angle']}° | {row['cells'][0]} × {row['cells'][1]} | {row['raw_valid']}/24 | {row['returned_valid']}/24 | {row['unique_valid']} | {row['median_rms_cells']:.3f} cells |\n"
    text += "\nReproduce with `python examples/geometry_optimization_study/feasible_geometry_screen.py`.\n"
    Path("docs/feasible_optimizer_geometry_screen.json").write_text(json.dumps(screen, indent=2))
Path("docs/feasible_optimizer_study.md").write_text(text, encoding="utf8")
Path("docs/feasible_optimizer_results.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
print(f"Published {len(rows)} measured cases")
