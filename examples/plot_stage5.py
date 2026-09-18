"""Plot Stage-5 distillation history, held-out errors and target diversity."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training = json.loads(Path(args.training).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))

    epochs = [row["epoch"] for row in training["history"]]
    train = [row["train_imitation"] for row in training["history"]]
    validation = [row["validation"]["loss"] for row in training["history"]]
    strategies = ["uniform", "heuristic", "imitation", "distilled"]
    errors = [100 * evaluation["summary"][name]["median_em_error"] for name in strategies]
    counts = Counter(sample["selected"] for sample in targets["samples"])
    selected = sorted(counts, key=lambda name: (-counts[name], name))

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].plot(epochs, train, marker="o", markersize=3, label="train")
    axes[0].plot(epochs, validation, marker="o", markersize=3, label="validation")
    best = int(np.argmin(validation))
    axes[0].scatter(epochs[best], validation[best], color="black", zorder=3, label="best")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Epoch", ylabel="CDF target loss", title="Physics-target distillation")
    axes[0].legend(frameon=True)

    colors = ["#8c8c8c", "#e69f00", "#56b4e9", "#009e73"]
    axes[1].bar(strategies, errors, color=colors)
    axes[1].set(ylabel="Median max EM error (%)", title="Untouched IID references")
    axes[1].tick_params(axis="x", rotation=20)
    for index, value in enumerate(errors):
        axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom")

    axes[2].barh(selected[::-1], [counts[name] for name in selected[::-1]], color="#0072b2")
    axes[2].set(xlabel="Selected scene-budget targets", title="Winning candidate families")
    axes[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
