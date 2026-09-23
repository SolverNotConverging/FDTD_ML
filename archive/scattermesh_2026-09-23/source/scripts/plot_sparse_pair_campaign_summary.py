#!/usr/bin/env python3
"""Plot the frozen sparse-pair low-budget headroom by material topology."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = (
    ("dielectric+dielectric", "Dielectric pair"),
    ("dielectric+pec", "Dielectric + PEC"),
    ("pec+pec", "PEC pair"),
)
COLORS = {32: "#356ab7", 48: "#e58b34"}


def plot(report_path, output_path):
    report = json.loads(Path(report_path).read_text())
    if report["decision"] != "passes_sparse_pair_headroom":
        raise ValueError("This plot requires a completed qualifying campaign report")
    rows = [row for row in report["groups"] if row["cells"] in (32, 48)]
    if len(rows) != report["low_budget"]["group_count"]:
        raise ValueError("Low-budget group count differs from report summary")

    fig, (distribution, wins) = plt.subplots(
        1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    rng = np.random.default_rng(20260922)
    for family_index, (family, _) in enumerate(FAMILIES):
        for budget, offset in ((32, -0.18), (48, 0.18)):
            values = np.array([
                row["improvement_over_uniform"] for row in rows
                if row["material_topology"] == family and row["cells"] == budget
            ])
            if not len(values):
                raise ValueError(f"No {budget}-cell cases for {family}")
            x = family_index + offset
            distribution.boxplot(
                values, positions=[x], widths=0.27, patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": COLORS[budget], "alpha": 0.28, "edgecolor": COLORS[budget]},
                medianprops={"color": COLORS[budget], "linewidth": 2},
                whiskerprops={"color": COLORS[budget]},
                capprops={"color": COLORS[budget]},
            )
            distribution.scatter(
                x + rng.uniform(-0.105, 0.105, len(values)), values,
                s=16, alpha=0.65, color=COLORS[budget], edgecolors="none",
                label=f"{budget}×{budget}" if family_index == 0 else None,
            )
    distribution.axhline(1.0, color="#454545", linewidth=1, linestyle=":")
    distribution.axhline(1.05, color="#aa3e3e", linewidth=1, linestyle="--")
    distribution.text(2.53, 1.05, "1.05× win", color="#aa3e3e", va="bottom", fontsize=9)
    distribution.set_xticks(range(3), [label for _, label in FAMILIES])
    distribution.set_xlim(-0.55, 2.67)
    distribution.set_yscale("log")
    distribution.set_ylabel("Uniform score / best candidate score (log scale)")
    distribution.set_title("Exact-budget mesh headroom")
    distribution.legend(frameon=False, loc="upper right")
    distribution.grid(axis="y", alpha=0.2)

    for index, (family, label) in enumerate(FAMILIES):
        summary = report["by_material_topology"][family]
        fraction = summary["meaningful_win_fraction"]
        wins.bar(index, fraction, color="#4479aa" if index < 2 else "#7a679d", width=0.64)
        wins.text(index, min(fraction + 0.025, 0.96),
                  f"{summary['meaningful_win_count']}/{summary['group_count']}",
                  ha="center", fontsize=10)
    wins.axhline(0.4, color="#aa3e3e", linewidth=1.2, linestyle="--",
                 label="Frozen material-stratum gate (40%)")
    wins.set_ylim(0, 1)
    wins.set_xticks(range(3), [label for _, label in FAMILIES], rotation=15, ha="right")
    wins.set_ylabel("Fraction with ≥1.05× improvement")
    wins.set_title("Meaningful wins across 32×32 and 48×48")
    wins.legend(frameon=False, loc="upper right", fontsize=8)
    wins.grid(axis="y", alpha=0.2)

    low = report["low_budget"]
    fig.suptitle("Qualified sparse-pair teacher search", fontsize=15, y=1.01)
    fig.text(
        0.5, -0.03,
        f"{low['meaningful_win_count']}/{low['group_count']} low-budget wins; "
        f"median improvement {low['median_improvement_over_uniform']:.3f}×. "
        "Score = complex far-field + 0.25 RCS-log loss, with soft Nt exponent 0.1.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("runs/sparse_pair_campaign_96/report.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/sparse_pair_campaign_96/headroom_summary.png"))
    args = parser.parse_args()
    print(plot(args.report, args.output))


if __name__ == "__main__":
    main()
