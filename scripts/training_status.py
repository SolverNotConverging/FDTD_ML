#!/usr/bin/env python3
"""Display the mixed-budget teacher/CNN workflow without importing PyTorch."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from campaign_status import duration

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/training_mixed_v7_3000"


def status(output):
    def read(name):
        path = output / name
        return json.loads(path.read_text()) if path.exists() else {}

    stage = read("stage.json")
    workflow = read("workflow.json")
    progress = read("teacher/progress.json")
    training = read("pretraining/training.json")
    failure = read("failure.json")
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"Training workflow — {stamp}",
        str(output),
        f"Stage: {stage.get('stage', 'initializing')}",
    ]
    if workflow:
        lines.append(
            f"Budget policy: {workflow.get('budget_policy', 'fixed')} | "
            f"CPU workers: {workflow['workers']} | GPU: {workflow['gpu']} | "
            f"sparse weight: {workflow['config'].get('sparse_sample_weight', 1):g}×"
        )
    if progress:
        completed, total = progress["completed"], progress["total"]
        lines.append(
            f"Teacher targets: {completed:,}/{total:,} ({100 * completed / max(1, total):.1f}%) | "
            f"feasible: {progress['feasible']:,} | failed: {progress['failures']:,}"
        )
        lines.append(
            f"Recorded teacher elapsed: {duration(progress['elapsed_seconds'])} | "
            f"last update: {duration(time.time() - progress['updated_unix'])} ago"
        )
    if training:
        history = training.get("history", [])
        if history:
            row = history[-1]
            lines.append(
                f"Epoch: {row['epoch']}/{training['config']['epochs']} | "
                f"train loss: {row['train_imitation']:.6g} | "
                f"validation: {row['validation']['loss']:.6g} | "
                f"best: {training['best_validation_loss']:.6g}"
            )
            for name, values in row["validation"].get("by_budget_group", {}).items():
                lines.append(
                    f"  {name}: validation loss {values['loss']:.6g} ({values['samples']:,} samples)"
                )
            lines.append(f"Epochs without improvement: {training.get('stale_epochs', 0)}")
    if failure and failure.get("time", 0) >= stage.get("updated_unix", 0):
        lines.append("WORKFLOW FAILURE: " + failure.get("error", str(failure)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    args = parser.parse_args()
    if args.watch is not None and not 1 <= args.watch < float("inf"):
        parser.error("--watch must be a finite interval of at least 1 second")
    if not args.output.is_dir():
        parser.error(f"Workflow not found: {args.output}")
    try:
        while True:
            print(status(args.output.resolve()), flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
            print("\n" + "─" * 72)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
