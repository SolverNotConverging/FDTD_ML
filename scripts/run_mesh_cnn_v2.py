#!/usr/bin/env python3
"""Profile, freeze, resume, train and evaluate the C0--C2 mesh-CNN pilot."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scattermesh.campaign_v2 import atomic_json, freeze_manifest, mark_unfinished, run_data_worker
from scattermesh.dataset_v2 import build_legacy_import, build_new_dataset
from scattermesh.evaluation_v2 import (
    compare_test_models,
    evaluate_checkpoint,
    render_test_figures,
    select_checkpoint,
    summarize_test,
)
from scattermesh.profile_v2 import run_profile
from scattermesh.training_v2 import train_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs_v2" / "c0_c2_pilot"
ARCHIVE_RUNS = ROOT / "archive" / "scattermesh_2026-09-23" / "runs"
LEGACY_DATASET = ARCHIVE_RUNS / "simple_factorial_exact_distillation_6916879" / "dataset.json"
LEGACY_CASES = ARCHIVE_RUNS / "simple_factorial_exact_candidate_full_6916879"


def _group(commands, names, output, deadline):
    processes = []
    for command, name in zip(commands, names):
        log = output / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a")
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        processes.append((process, stream, name))
    try:
        while any(process.poll() is None for process, _, _ in processes):
            if time.time() >= deadline:
                for process, _, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                for process, _, _ in processes:
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                break
            time.sleep(3)
    finally:
        for _, stream, _ in processes:
            stream.close()
    return {name: process.returncode for process, _, name in processes}


def launch(output):
    """Enforce phase deadlines; stop before bulk if the sizing gate fails."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start_path = output / "campaign_start.json"
    if start_path.exists():
        started = json.loads(start_path.read_text())["started_epoch_s"]
    else:
        started = time.time()
        atomic_json(
            start_path, {"started_epoch_s": started, "deadline_epoch_s": started + 24 * 3600}
        )
    profile_path = output / "profile" / "profile.json"
    if not profile_path.exists():
        run_profile(profile_path.parent)
    profile = json.loads(profile_path.read_text())
    if profile["gate"] != "ready_for_bulk":
        status = {
            "phase": "sizing_gate_stopped",
            "reason": "Neither balanced dataset is established within 14-hour data allocation",
            "profile": str(profile_path),
            "partial_projected_data_hours_by_lineages": profile.get(
                "partial_projected_data_hours_by_lineages"
            ),
        }
        atomic_json(output / "campaign_status.json", status)
        return status
    manifest_path = output / "manifest.json"
    freeze_manifest(profile_path, manifest_path)
    legacy_path = output / "datasets" / "legacy" / "dataset.json"
    if not legacy_path.exists():
        build_legacy_import(LEGACY_DATASET, LEGACY_CASES, legacy_path.parent)
    data_deadline = started + 15 * 3600
    commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "data-worker",
            "--output",
            str(output),
            "--shard",
            str(index),
            "--device",
            f"cuda:{index}",
            "--deadline",
            str(data_deadline),
        ]
        for index in range(4)
    ]
    codes = _group(commands, [f"data_gpu{index}" for index in range(4)], output, data_deadline)
    incomplete = mark_unfinished(manifest_path, output)
    atomic_json(
        output / "phase_data.json",
        {"return_codes": codes, "ended_epoch_s": time.time(), "incomplete": incomplete},
    )
    try:
        build_new_dataset(manifest_path, output, output / "datasets" / "new")
    except ValueError as error:
        status = {"phase": "data_incomplete", "error": str(error)}
        atomic_json(output / "campaign_status.json", status)
        return status
    training_deadline = started + 21 * 3600
    model_names = (("large", 64, "cuda:0"), ("small", 16, "cuda:1"))
    commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "train",
            "--output",
            str(output),
            "--name",
            name,
            "--base",
            str(base),
            "--device",
            device,
            "--deadline",
            str(training_deadline),
        ]
        for name, base, device in model_names
    ]
    codes = _group(commands, [f"train_{row[0]}" for row in model_names], output, training_deadline)
    atomic_json(
        output / "phase_training.json", {"return_codes": codes, "ended_epoch_s": time.time()}
    )
    checkpoints_by_model = {
        name: sorted((output / "training" / name).glob("checkpoint_epoch_*.pt"))
        for name, _, _ in model_names
    }
    for index in range(3):
        commands, labels = [], []
        for name, _, device in model_names:
            if len(checkpoints_by_model[name]) != 3:
                continue
            commands.append(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "evaluate",
                    "--output",
                    str(output),
                    "--split",
                    "validation",
                    "--name",
                    name,
                    "--eval-id",
                    f"candidate_{index}",
                    "--checkpoint",
                    str(checkpoints_by_model[name][index]),
                    "--device",
                    device,
                    "--deadline",
                    str(training_deadline),
                ]
            )
            labels.append(f"validation_{name}_{index}")
        if commands and time.time() < training_deadline:
            _group(commands, labels, output, training_deadline)
    selections = {}
    for name, _, _ in model_names:
        paths = [
            output / "evaluation" / "validation" / name / f"candidate_{index}" / "summary.json"
            for index in range(3)
        ]
        if all(path.exists() for path in paths):
            try:
                selections[name] = select_checkpoint(
                    paths, output / "training" / name / "frozen_selection.json"
                )
            except ValueError:
                pass
    if len(selections) != 2:
        status = {"phase": "validation_incomplete", "selected_models": list(selections)}
        atomic_json(output / "campaign_status.json", status)
        return status
    test_deadline = started + 24 * 3600
    commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "evaluate",
            "--output",
            str(output),
            "--split",
            "test",
            "--name",
            name,
            "--checkpoint",
            selections[name]["checkpoint"],
            "--device",
            device,
            "--deadline",
            str(test_deadline),
        ]
        for name, _, device in model_names
    ]
    if time.time() < test_deadline:
        _group(commands, [f"test_{name}" for name, _, _ in model_names], output, test_deadline)
    for name, _, _ in model_names:
        evaluation_path = output / "evaluation" / "test" / name
        if not (evaluation_path / "summary.json").exists():
            status = {"phase": "test_incomplete", "reason": f"No test summary for {name}"}
            atomic_json(output / "campaign_status.json", status)
            return status
        summarize_test(
            evaluation_path / "summary.json", output, output / "reports" / f"{name}_test.json"
        )
        render_test_figures(
            output / "reports" / f"{name}_test.json",
            manifest_path,
            output,
            evaluation_path,
            output / "reports" / f"{name}_figures",
        )
    comparison = compare_test_models(
        output / "reports" / "large_test.json",
        output / "reports" / "small_test.json",
        output / "reports" / "large_vs_small.json",
    )
    status = {
        "phase": "completed"
        if comparison["large_test_complete"] and comparison["small_test_complete"]
        else "test_incomplete",
        "ended_epoch_s": time.time(),
        "deadline_epoch_s": test_deadline,
        "scientific_gate": json.loads((output / "reports" / "large_test.json").read_text())[
            "positive_dielectric_gate"
        ],
    }
    atomic_json(output / "campaign_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "profile",
            "freeze",
            "data-worker",
            "legacy-import",
            "targets",
            "train",
            "evaluate",
            "select",
            "report",
            "launch",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--name", default="large")
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--eval-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--summaries", type=Path, nargs="*")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if args.command == "profile":
        result = run_profile(output / "profile")
    elif args.command == "freeze":
        result = {
            "lineage_count": freeze_manifest(output / "profile" / "profile.json", manifest)[
                "lineage_count"
            ]
        }
    elif args.command == "data-worker":
        result = run_data_worker(
            manifest, output, shard=args.shard, device=args.device, deadline=args.deadline
        )
    elif args.command == "legacy-import":
        result = {
            "dataset": str(
                build_legacy_import(LEGACY_DATASET, LEGACY_CASES, output / "datasets" / "legacy")
            )
        }
    elif args.command == "targets":
        result = {"dataset": str(build_new_dataset(manifest, output, output / "datasets" / "new"))}
    elif args.command == "train":
        result = train_model(
            output / "datasets" / "new" / "dataset.json",
            output / "datasets" / "legacy" / "dataset.json",
            output / "training" / args.name,
            base_channels=args.base,
            device=args.device,
            deadline=args.deadline,
            microbatch=args.microbatch,
        )
    elif args.command == "evaluate":
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        result = evaluate_checkpoint(
            manifest,
            output,
            args.checkpoint,
            output / "evaluation" / args.split / args.name / args.eval_id
            if args.eval_id
            else output / "evaluation" / args.split / args.name,
            split=args.split,
            device=args.device,
            deadline=args.deadline,
        )
        result = {
            key: result[key]
            for key in (
                "split",
                "requested_conditions",
                "valid_conditions",
                "mean_learned_raw_accuracy_loss",
                "terminal",
            )
        }
    elif args.command == "select":
        if not args.summaries:
            parser.error("select requires three --summaries")
        result = select_checkpoint(
            args.summaries, output / "training" / args.name / "frozen_selection.json"
        )
    elif args.command == "report":
        result = summarize_test(
            output / "evaluation" / "test" / args.name / "summary.json",
            output,
            output / "reports" / f"{args.name}_test.json",
        )
        result = {
            key: result[key]
            for key in (
                "test_complete",
                "positive_dielectric_gate",
                "requested_test_conditions",
                "valid_test_conditions",
            )
        }
    else:
        result = launch(output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
