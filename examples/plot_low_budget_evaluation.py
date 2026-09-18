"""Plot per-budget validation or test metrics from a Stage-5 evaluation report."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "distilled": "Low-budget continuation",
    "imitation": "Prior Stage 5",
    "heuristic": "Heuristic",
    "uniform": "Uniform",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Held-out FDTD accuracy by mesh budget")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    values = defaultdict(list)
    for row in report["rows"]:
        if row["status"] != "ok":
            continue
        error = max(row["metrics"]["waveform_l2_max"], row["metrics"]["spectrum_l2_max"])
        values[row["strategy"], row["budget"][0]].append(100 * error)

    budgets = sorted({budget for _, budget in values})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for strategy in ("distilled", "imitation", "heuristic", "uniform"):
        medians = [np.median(values[strategy, budget]) for budget in budgets]
        means = [np.mean(values[strategy, budget]) for budget in budgets]
        style = (
            {"lw": 2.4, "marker": "o"} if strategy == "distilled" else {"lw": 1.5, "marker": "o"}
        )
        axes[0].plot(budgets, medians, label=LABELS[strategy], **style)
        axes[1].plot(budgets, means, label=LABELS[strategy], **style)
    for ax, statistic in zip(axes, ("Median", "Mean")):
        ax.set(
            xlabel="Exact cells per axis",
            ylabel=f"{statistic} max EM error (%)",
            xticks=budgets,
            yscale="log",
        )
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(args.title, fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
