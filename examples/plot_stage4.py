"""Plot Stage-4 imitation learning curves and held-out projection corrections."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.training.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    history = training["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train_imitation"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation"]["loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="CDF imitation MSE", yscale="log")
    axes[0].legend()
    metrics = evaluation["imitation"]
    labels = ["x mean", "x max", "y mean", "y max"]
    values = [
        metrics["x_projection_l1_mean"],
        metrics["x_projection_l1_max"],
        metrics["y_projection_l1_mean"],
        metrics["y_projection_l1_max"],
    ]
    axes[1].bar(labels, values, color=["#3274ad", "#3274ad", "#d65f32", "#d65f32"])
    axes[1].set(ylabel="Normalized mean line correction", title="Held-out deterministic repair")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Stage 4: heuristic-teacher imitation")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
