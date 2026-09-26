#!/usr/bin/env python3
"""Profile, freeze, resume, train and evaluate mesh-CNN curriculum campaigns."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from scattermesh.campaign_v2 import (
    atomic_json,
    freeze_compact_manifest,
    freeze_manifest,
    mark_unfinished,
    run_data_worker,
)
from scattermesh.dataset_v2 import build_legacy_import, build_new_dataset
from scattermesh.evaluation_v2 import (
    compare_test_models,
    evaluate_checkpoint,
    render_test_figures,
    select_checkpoint,
    summarize_test,
)
from scattermesh.profile_poc import run_compact_profile
from scattermesh.profile_v2 import run_profile
from scattermesh.training_v2 import train_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs_v2" / "c0_c2_pilot"
DEFAULT_COMPACT_OUTPUT = ROOT / "runs_v2" / "compact_poc"
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


def launch_compact(output, *, seed=20260924, max_campaign_hours=24.0):
    """Run the frozen large-model C8/C9 proof-of-concept under measured phase limits."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not 0 < max_campaign_hours <= 24:
        raise ValueError("The compact campaign deadline must be in (0, 24] hours")

    profile_path = output / "profile" / "compact_profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
    else:
        profile = run_compact_profile(
            profile_path.parent, seed=seed, max_campaign_hours=max_campaign_hours
        )
    if profile.get("seed") != seed:
        status = {
            "phase": "development_profile_stopped",
            "reason": "Saved compact profile seed differs from the requested campaign seed",
            "profile": str(profile_path),
        }
        atomic_json(output / "campaign_status.json", status)
        return status
    if profile.get("gate") != "ready_for_compact_campaign":
        status = {
            "phase": "development_profile_stopped",
            "reason": profile.get("status", profile.get("gate", "profile_not_ready")),
            "profile": str(profile_path),
            "estimated_compact_campaign_hours": profile.get("estimated_compact_campaign_hours"),
        }
        atomic_json(output / "campaign_status.json", status)
        return status

    estimate = profile.get("estimated_compact_campaign_hours")
    if not isinstance(estimate, (int, float)) or estimate > max_campaign_hours:
        status = {
            "phase": "development_profile_stopped",
            "reason": "Measured profile exceeds the requested campaign deadline",
            "estimated_compact_campaign_hours": estimate,
            "max_campaign_hours": max_campaign_hours,
        }
        atomic_json(output / "campaign_status.json", status)
        return status

    import torch

    profiled_workers = int(profile.get("estimated_data_worker_gpu_count", 0))
    current_workers = torch.cuda.device_count() if torch.cuda.is_available() else 0
    workers = min(4, profiled_workers, current_workers)
    if workers < 1:
        status = {
            "phase": "hardware_stopped",
            "reason": "No CUDA worker is available for the measured compact campaign",
            "profiled_data_worker_gpu_count": profiled_workers,
            "current_cuda_device_count": current_workers,
        }
        atomic_json(output / "campaign_status.json", status)
        return status

    manifest_path = output / "manifest.json"
    try:
        manifest = freeze_compact_manifest(profile_path, manifest_path, seed=seed)
    except (ValueError, OSError) as error:
        status = {
            "phase": "manifest_stopped",
            "reason": f"{type(error).__name__}: {error}",
        }
        atomic_json(output / "campaign_status.json", status)
        return status

    phase_wall = profile.get("estimated_fdt_wall_hours_by_phase", {})
    training_hours = profile.get("estimated_training_hours")
    margin = float(profile.get("cost_model_margin", 1.35))
    required = ("data", "validation", "test")
    if (
        any(not isinstance(phase_wall.get(name), (int, float)) for name in required)
        or not isinstance(training_hours, (int, float))
        or any(phase_wall[name] < 0 for name in required)
        or training_hours < 0
    ):
        status = {
            "phase": "profile_stopped",
            "reason": "The measured profile lacks phase-specific runtime estimates",
        }
        atomic_json(output / "campaign_status.json", status)
        return status
    allocations = {
        "data": phase_wall["data"] * margin,
        "training": training_hours * margin,
        "validation": phase_wall["validation"] * margin,
        "test": phase_wall["test"] * margin,
        "overhead": 0.5 * margin,
    }
    allocated_hours = sum(allocations.values())
    if allocated_hours > max_campaign_hours:
        status = {
            "phase": "profile_stopped",
            "reason": "Phase allocations exceed the requested wall-clock deadline",
            "phase_allocations_hours": allocations,
            "allocated_hours": allocated_hours,
            "max_campaign_hours": max_campaign_hours,
        }
        atomic_json(output / "campaign_status.json", status)
        return status

    start_path = output / "campaign_start.json"
    profile_hash = (
        profile.get("source_profile_sha256")
        or hashlib.sha256(profile_path.read_bytes()).hexdigest()
    )
    if start_path.exists():
        start = json.loads(start_path.read_text())
        if start.get("seed") != seed or start.get("profile_sha256") != profile_hash:
            raise ValueError("Cannot resume a compact campaign with a different seed or profile")
        started = float(start["started_epoch_s"])
    else:
        started = time.time()
        atomic_json(
            start_path,
            {
                "started_epoch_s": started,
                "deadline_epoch_s": started + max_campaign_hours * 3600,
                "seed": seed,
                "profile_sha256": profile_hash,
                "phase_allocations_hours": allocations,
            },
        )
    deadline = started + max_campaign_hours * 3600
    data_deadline = min(deadline, started + allocations["data"] * 3600)
    training_deadline = min(deadline, data_deadline + allocations["training"] * 3600)
    validation_deadline = min(deadline, training_deadline + allocations["validation"] * 3600)

    legacy_path = output / "datasets" / "legacy" / "dataset.json"
    if not legacy_path.exists():
        try:
            build_legacy_import(LEGACY_DATASET, LEGACY_CASES, legacy_path.parent)
        except (ValueError, OSError) as error:
            status = {
                "phase": "legacy_import_stopped",
                "reason": f"{type(error).__name__}: {error}",
            }
            atomic_json(output / "campaign_status.json", status)
            return status

    worker_commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "data-worker",
            "--output",
            str(output),
            "--shard",
            str(index),
            "--shards",
            str(workers),
            "--device",
            f"cuda:{index}",
            "--deadline",
            str(data_deadline),
        ]
        for index in range(workers)
    ]
    worker_codes = _group(
        worker_commands,
        [f"compact_data_gpu{index}" for index in range(workers)],
        output,
        data_deadline,
    )
    incomplete = mark_unfinished(manifest_path, output)
    atomic_json(
        output / "phase_data.json",
        {
            "return_codes": worker_codes,
            "ended_epoch_s": time.time(),
            "incomplete": incomplete,
        },
    )
    try:
        dataset_path = build_new_dataset(manifest_path, output, output / "datasets" / "new")
        dataset = json.loads(dataset_path.read_text())
        expected_examples = sum(
            len(conditions) for conditions in manifest["conditions_by_scene"].values()
        )
        if len(dataset["examples"]) != expected_examples:
            raise ValueError(
                f"Expected {expected_examples} complete train/validation conditions; "
                f"found {len(dataset['examples'])}"
            )
    except (ValueError, OSError) as error:
        status = {"phase": "data_incomplete", "reason": f"{type(error).__name__}: {error}"}
        atomic_json(output / "campaign_status.json", status)
        return status

    train_codes = _group(
        [
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "train",
                "--output",
                str(output),
                "--name",
                "large",
                "--base",
                "64",
                "--device",
                "cuda:0",
                "--microbatch",
                str(profile["training_benchmark"]["microbatch"]),
                "--max-epochs",
                "60",
                "--deadline",
                str(training_deadline),
            ]
        ],
        ["compact_train_large"],
        output,
        training_deadline,
    )
    atomic_json(
        output / "phase_training.json",
        {"return_codes": train_codes, "ended_epoch_s": time.time()},
    )
    checkpoints = list((output / "training" / "large").glob("checkpoint_epoch_*.pt"))
    checkpoints.sort(
        key=lambda path: torch.load(path, map_location="cpu", weights_only=False)[
            "validation_profile_loss"
        ]
    )
    if train_codes.get("compact_train_large") != 0 or len(checkpoints) != 3:
        status = {
            "phase": "training_incomplete",
            "return_codes": train_codes,
            "available_checkpoints": len(checkpoints),
        }
        atomic_json(output / "campaign_status.json", status)
        return status

    validation_summaries = []
    for index, checkpoint in enumerate(checkpoints):
        code = _group(
            [
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "evaluate",
                    "--output",
                    str(output),
                    "--split",
                    "validation",
                    "--name",
                    "large",
                    "--eval-id",
                    f"candidate_{index}",
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cuda:0",
                    "--deadline",
                    str(validation_deadline),
                ]
            ],
            [f"compact_validation_{index}"],
            output,
            validation_deadline,
        )
        summary = (
            output / "evaluation" / "validation" / "large" / f"candidate_{index}" / "summary.json"
        )
        if code.get(f"compact_validation_{index}") == 0 and summary.exists():
            validation_summaries.append(summary)
        else:
            break
    if len(validation_summaries) != 3:
        status = {
            "phase": "validation_incomplete",
            "available_validation_summaries": len(validation_summaries),
        }
        atomic_json(output / "campaign_status.json", status)
        return status
    selection = select_checkpoint(
        validation_summaries, output / "training" / "large" / "frozen_selection.json"
    )

    test_deadline = min(deadline, validation_deadline + allocations["test"] * 3600)
    test_codes = _group(
        [
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "evaluate",
                "--output",
                str(output),
                "--split",
                "test",
                "--name",
                "large",
                "--checkpoint",
                selection["checkpoint"],
                "--device",
                "cuda:0",
                "--deadline",
                str(test_deadline),
            ]
        ],
        ["compact_test_large"],
        output,
        test_deadline,
    )
    test_summary_path = output / "evaluation" / "test" / "large" / "summary.json"
    if test_codes.get("compact_test_large") != 0 or not test_summary_path.exists():
        status = {"phase": "test_incomplete", "return_codes": test_codes}
        atomic_json(output / "campaign_status.json", status)
        return status
    report = summarize_test(test_summary_path, output, output / "reports" / "large_test.json")
    render_test_figures(
        output / "reports" / "large_test.json",
        manifest_path,
        output,
        output / "evaluation" / "test" / "large",
        output / "reports" / "large_figures",
    )
    status = {
        "phase": "completed" if report["test_complete"] else "test_incomplete",
        "ended_epoch_s": time.time(),
        "deadline_epoch_s": deadline,
        "positive_demonstration": report["positive_demonstration"],
        "requested_test_conditions": report["requested_test_conditions"],
        "valid_test_conditions": report["valid_test_conditions"],
        "model_comparison": "large_model_only; small-model ablation deferred",
    }
    atomic_json(output / "campaign_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "profile",
            "profile-compact",
            "freeze",
            "freeze-compact",
            "data-worker",
            "legacy-import",
            "targets",
            "train",
            "evaluate",
            "select",
            "report",
            "launch",
            "launch-compact",
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--name", default="large")
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--eval-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--summaries", type=Path, nargs="*")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-campaign-hours", type=float, default=24.0)
    args = parser.parse_args()
    if args.command in ("data-worker", "evaluate") and not args.device.startswith("cuda"):
        parser.error("New FDTD campaign runs require the compiled CUDA kernel")
    selected_output = args.output or (
        DEFAULT_COMPACT_OUTPUT
        if args.command in ("profile-compact", "freeze-compact", "launch-compact")
        else DEFAULT_OUTPUT
    )
    output = selected_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if args.command == "profile":
        result = run_profile(output / "profile")
    elif args.command == "profile-compact":
        result = run_compact_profile(
            output / "profile",
            seed=args.seed,
            max_campaign_hours=args.max_campaign_hours,
        )
    elif args.command == "freeze":
        result = {
            "lineage_count": freeze_manifest(output / "profile" / "profile.json", manifest)[
                "lineage_count"
            ]
        }
    elif args.command == "freeze-compact":
        result = {
            "lineage_count": freeze_compact_manifest(
                output / "profile" / "compact_profile.json", manifest, seed=args.seed
            )["lineage_count"]
        }
    elif args.command == "data-worker":
        result = run_data_worker(
            manifest,
            output,
            shard=args.shard,
            shards=args.shards,
            device=args.device,
            deadline=args.deadline,
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
            max_epochs=args.max_epochs,
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
    elif args.command == "launch-compact":
        result = launch_compact(output, seed=args.seed, max_campaign_hours=args.max_campaign_hours)
    else:
        result = launch(output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
