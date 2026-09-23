#!/usr/bin/env python3
"""Adaptively escalate dielectric references until qualified or at a hard limit."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = ROOT / "scripts/run_multi_gpu_campaign.py"
SPEC = importlib.util.spec_from_file_location("scattermesh_multi_gpu_campaign", SCHEDULER_PATH)
SCHEDULER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCHEDULER)
TERMINAL = {"accepted", "skipped"}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_policy(raw, path):
    if raw.get("schema_version") != 1 or not isinstance(raw.get("campaign_id"), str):
        raise ValueError("Policy requires schema_version 1 and a campaign_id")
    threshold = raw.get("variation_threshold", 0.005)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0:
        raise ValueError("variation_threshold must be positive")
    output_root = Path(raw.get("output_root", "runs/adaptive_references"))
    if not output_root.is_absolute():
        output_root = (path.parent / output_root).resolve()
    scenes, identifiers = [], set()
    for raw_scene in raw.get("scenes", []):
        scene_id, scene = raw_scene.get("scene_id"), raw_scene.get("scene")
        if (
            not isinstance(scene_id, str)
            or not SCHEDULER.TASK_ID.fullmatch(scene_id)
            or scene_id in identifiers
        ):
            raise ValueError("Every policy scene needs a unique nonempty scene_id")
        identifiers.add(scene_id)
        if scene not in SCHEDULER.SCENES:
            raise ValueError(f"Unsupported runner scene {scene}")
        cells = raw_scene.get("cell_levels")
        durations = raw_scene.get("duration_levels_ns")
        samples = raw_scene.get("sample_levels", [24, 48])
        if (
            not isinstance(cells, list)
            or len(cells) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 16
                for value in cells
            )
            or cells != sorted(set(cells))
            or not isinstance(durations, list)
            or len(durations) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
                for value in durations
            )
            or durations != sorted(set(durations))
            or not isinstance(samples, list)
            or len(samples) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in samples
            )
            or samples != sorted(set(samples))
        ):
            raise ValueError(f"Scene {scene_id} requires strictly increasing level lists")
        pml = raw_scene.get("pml", 0.15)
        if isinstance(pml, bool) or not isinstance(pml, (int, float)) or pml <= 0:
            raise ValueError(f"Scene {scene_id} has invalid PML thickness")
        scenes.append(
            dict(
                scene_id=scene_id,
                scene=scene,
                cell_levels=cells,
                duration_levels_ns=[float(value) for value in durations],
                sample_levels=samples,
                pml=float(pml),
            )
        )
    if not scenes:
        raise ValueError("Policy requires at least one scene")
    return dict(
        schema_version=1,
        campaign_id=raw["campaign_id"],
        output_root=str(output_root),
        variation_threshold=float(threshold),
        scenes=scenes,
    )


def initial_state(policy, digest):
    return dict(
        schema_version=1,
        campaign_id=policy["campaign_id"],
        policy_sha256=digest,
        status="running",
        started_unix=time.time(),
        updated_unix=time.time(),
        wave_count=0,
        attempts={},
        scenes={
            scene["scene_id"]: dict(
                decision="pending",
                cell_index=0,
                duration_index=0,
                sample_index=0,
                escalation_history=[],
            )
            for scene in policy["scenes"]
        },
    )


def load_state(path, policy, digest):
    if not path.exists():
        return initial_state(policy, digest)
    state = json.loads(path.read_text())
    if state.get("policy_sha256") != digest or state.get("campaign_id") != policy["campaign_id"]:
        raise ValueError("Convergence state belongs to a different immutable policy")
    state["status"] = "running"
    return state


def attempt_config(scene, scene_state, *, kind="base"):
    cell_index = scene_state["cell_index"] + (1 if kind == "spatial" else 0)
    duration_index = scene_state["duration_index"] + (1 if kind == "duration" else 0)
    sample_index = scene_state["sample_index"] + (1 if kind == "quadrature" else 0)
    return dict(
        scene=scene["scene"],
        cells=scene["cell_levels"][cell_index],
        duration_ns=scene["duration_levels_ns"][duration_index],
        samples=scene["sample_levels"][sample_index],
        pml=scene["pml"],
        larger_contour=kind == "contour",
    )


def attempt_key(scene_id, config):
    return f"{scene_id}:{sha256_json(config)}"


def load_attempt(state, scene_id, config):
    row = state["attempts"].get(attempt_key(scene_id, config))
    if row is None or row.get("scheduler_status") not in {"succeeded", "nonconverged"}:
        return None
    record_path = Path(row["result_path"])
    if not record_path.exists() or not record_path.with_name("spectra.npz").exists():
        row["scheduler_status"] = "invalid"
        return None
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        row["scheduler_status"] = "invalid"
        return None
    if not SCHEDULER.task_matches(
        record, row["task"], SCHEDULER.source_sha256()
    ) or not SCHEDULER.valid_artifact(record_path, record):
        row["scheduler_status"] = "invalid"
        return None
    with np.load(record_path.with_name("spectra.npz")) as arrays:
        numerical = arrays["complex_numerical"].copy()
        analytic = arrays["complex_analytic"].copy()
    return dict(record=record, numerical=numerical, analytic=analytic, path=str(record_path))


def promote(scene, scene_state, dimensions, reason):
    changes = {}
    for dimension in sorted(set(dimensions)):
        index_name = f"{dimension}_index"
        levels_name = {
            "cell": "cell_levels",
            "duration": "duration_levels_ns",
            "sample": "sample_levels",
        }[dimension]
        current = scene_state[index_name]
        if current + 1 >= len(scene[levels_name]):
            scene_state.update(
                decision="skipped",
                reason="hard_limit",
                hard_limit_dimension=dimension,
                hard_limit_reason=reason,
                completed_unix=time.time(),
            )
            return False
        scene_state[index_name] = current + 1
        changes[dimension] = dict(
            before=scene[levels_name][current], after=scene[levels_name][current + 1]
        )
    scene_state["escalation_history"].append(
        dict(reason=reason, changes=changes, time_unix=time.time())
    )
    return True


def field_change(candidate, base):
    scale = np.maximum(np.linalg.norm(base["analytic"], axis=1), 1e-30)
    values = np.linalg.norm(candidate["numerical"] - base["numerical"], axis=1) / scale
    return dict(by_frequency=values.tolist(), maximum=float(values.max()))


def evaluate_scene(scene, scene_state, state, threshold):
    """Advance one scene until it needs attempts or reaches a terminal decision."""
    while scene_state["decision"] not in TERMINAL:
        base_config = attempt_config(scene, scene_state)
        base = load_attempt(state, scene["scene_id"], base_config)
        if base is None:
            return [("base", base_config)]
        record = base["record"]
        if record.get("status") != "pass":
            failed = {name for name, gate in record.get("gates", {}).items() if not gate["passed"]}
            if "max_analytic_series_relative_l2" in failed or not record.get(
                "finite_fields", False
            ):
                scene_state.update(
                    decision="skipped",
                    reason="unrecoverable_individual_gate",
                    failed_gates=sorted(failed),
                    completed_unix=time.time(),
                )
                continue
            dimensions = []
            if "tail_peak_over_global_peak" in failed:
                dimensions.append("duration")
            if failed & {"max_complex_relative_l2", "max_phase_rms_degrees"}:
                dimensions.append("cell")
            if not dimensions:
                scene_state.update(
                    decision="skipped",
                    reason="unclassified_individual_failure",
                    failed_gates=sorted(failed),
                    completed_unix=time.time(),
                )
                continue
            promote(scene, scene_state, dimensions, "individual_gate_failure")
            continue

        if scene_state["cell_index"] + 1 >= len(scene["cell_levels"]):
            promote(scene, scene_state, ["cell"], "spatial_probe_unavailable")
            continue
        if scene_state["duration_index"] + 1 >= len(scene["duration_levels_ns"]):
            promote(scene, scene_state, ["duration"], "duration_probe_unavailable")
            continue
        if scene_state["sample_index"] + 1 >= len(scene["sample_levels"]):
            promote(scene, scene_state, ["sample"], "quadrature_probe_unavailable")
            continue

        probes = {
            kind: attempt_config(scene, scene_state, kind=kind)
            for kind in ("spatial", "duration", "quadrature", "contour")
        }
        loaded = {
            kind: load_attempt(state, scene["scene_id"], config) for kind, config in probes.items()
        }
        missing = [(kind, probes[kind]) for kind in probes if loaded[kind] is None]
        if missing:
            return missing

        failed_probe_dimensions = []
        for kind, attempt in loaded.items():
            if attempt["record"].get("status") != "pass":
                failed_probe_dimensions.append(
                    {
                        "spatial": "cell",
                        "duration": "duration",
                        "quadrature": "sample",
                        "contour": "cell",
                    }[kind]
                )
        if failed_probe_dimensions:
            promote(scene, scene_state, failed_probe_dimensions, "probe_individual_gate_failure")
            continue

        variations = {kind: field_change(attempt, base) for kind, attempt in loaded.items()}
        failed_variations = [
            kind for kind, value in variations.items() if value["maximum"] >= threshold
        ]
        if failed_variations:
            dimensions = [
                {
                    "spatial": "cell",
                    "duration": "duration",
                    "quadrature": "sample",
                    "contour": "cell",
                }[kind]
                for kind in failed_variations
            ]
            promote(scene, scene_state, dimensions, "independent_variation_failure")
            continue

        scene_state.update(
            decision="accepted",
            completed_unix=time.time(),
            accepted_config=base_config,
            accepted_record=base["path"],
            probes={kind: attempt["path"] for kind, attempt in loaded.items()},
            complex_field_variations=variations,
            variation_threshold=threshold,
        )
    return []


def make_tasks(missing, state, devices):
    tasks = []
    for scene_id, kind, config in missing:
        key = attempt_key(scene_id, config)
        existing = state["attempts"].get(key)
        if existing is not None and existing.get("scheduler_status") != "invalid":
            continue
        digest = sha256_json(dict(scene_id=scene_id, config=config))
        device = devices[int(digest[:12], 16) % len(devices)]
        task = dict(
            task_id=f"{scene_id}_{kind}_{digest[:12]}",
            **config,
            device=device,
        )
        invalidations = 0 if existing is None else existing.get("invalidations", 0) + 1
        total_process_attempts = (
            0 if existing is None else existing.get("total_process_attempts", 0)
        )
        state["attempts"][key] = dict(
            scene_id=scene_id,
            kind=kind,
            config=config,
            task=task,
            scheduler_status="planned",
            invalidations=invalidations,
            total_process_attempts=total_process_attempts,
        )
        tasks.append(task)
    return tasks


def consume_wave(state, wave_state):
    by_task = {row["task"]["task_id"]: row for row in state["attempts"].values()}
    for task_id, result in wave_state["tasks"].items():
        row = by_task[task_id]
        row["scheduler_status"] = result["status"]
        row["attempts"] = result["attempts"]
        row["total_process_attempts"] = row.get("total_process_attempts", 0) + result["attempts"]
        if "result_path" in result:
            row["result_path"] = result["result_path"]
            row["result_fingerprint"] = result.get("result_fingerprint")
        if result["status"] == "failed":
            scene_state = state["scenes"][row["scene_id"]]
            scene_state.update(
                decision="skipped",
                reason="execution_failure",
                failed_task=task_id,
                completed_unix=time.time(),
            )


def summary(state):
    counts = Counter(row["decision"] for row in state["scenes"].values())
    return dict(
        campaign_id=state["campaign_id"],
        status=state["status"],
        scene_counts=dict(sorted(counts.items())),
        attempt_count=len(state["attempts"]),
        wave_count=state["wave_count"],
        started_unix=state["started_unix"],
        updated_unix=state["updated_unix"],
        scenes=state["scenes"],
    )


def revalidate_accepted(policy, state):
    for scene in policy["scenes"]:
        scene_state = state["scenes"][scene["scene_id"]]
        if scene_state["decision"] != "accepted":
            continue
        configs = [attempt_config(scene, scene_state)] + [
            attempt_config(scene, scene_state, kind=kind)
            for kind in ("spatial", "duration", "quadrature", "contour")
        ]
        if all(load_attempt(state, scene["scene_id"], config) is not None for config in configs):
            continue
        scene_state["decision"] = "pending"
        scene_state["acceptance_invalidated_unix"] = time.time()
        for key in (
            "completed_unix",
            "accepted_config",
            "accepted_record",
            "probes",
            "complex_field_variations",
            "variation_threshold",
        ):
            scene_state.pop(key, None)


def run_policy(policy, output, devices, *, runner=SCHEDULER.DEFAULT_RUNNER, max_attempts=2):
    digest = sha256_json(policy)
    state_path = output / "state.json"
    state = load_state(state_path, policy, digest)
    revalidate_accepted(policy, state)
    output.mkdir(parents=True, exist_ok=True)
    while True:
        missing = []
        for scene in policy["scenes"]:
            scene_state = state["scenes"][scene["scene_id"]]
            for kind, config in evaluate_scene(
                scene, scene_state, state, policy["variation_threshold"]
            ):
                missing.append((scene["scene_id"], kind, config))
        if all(row["decision"] in TERMINAL for row in state["scenes"].values()):
            break
        tasks = make_tasks(missing, state, devices)
        if not tasks:
            raise RuntimeError("Policy made no progress and has no new attempts")
        wave_index = state["wave_count"]
        wave_directory = output / "waves" / f"wave{wave_index:03d}"
        wave_plan = dict(
            schema_version=1,
            campaign_id=f"{policy['campaign_id']}_wave{wave_index:03d}",
            output_root=policy["output_root"],
            tasks=tasks,
        )
        wave_directory.mkdir(parents=True, exist_ok=True)
        plan_path = wave_directory / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != wave_plan:
            raise ValueError(f"Existing wave plan differs at {plan_path}")
        atomic_json(plan_path, wave_plan)
        validated = SCHEDULER.validate_plan(wave_plan, plan_path, devices)
        wave_state = SCHEDULER.run_campaign(
            validated,
            wave_directory / "coordinator",
            runner=runner,
            max_attempts=max_attempts,
        )
        consume_wave(state, wave_state)
        state["wave_count"] += 1
        state["updated_unix"] = time.time()
        atomic_json(state_path, state)

    counts = Counter(row["decision"] for row in state["scenes"].values())
    state["status"] = "complete" if not counts.get("pending", 0) else "running"
    state["completed_unix"] = time.time()
    state["updated_unix"] = time.time()
    atomic_json(state_path, state)
    atomic_json(output / "summary.json", summary(state))
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--runner", type=Path, default=SCHEDULER.DEFAULT_RUNNER, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()
    if args.status:
        state = json.loads((output / "state.json").read_text())
        print(json.dumps(summary(state), indent=2))
        return
    if (
        not args.devices
        or len(set(args.devices)) != len(args.devices)
        or any(not SCHEDULER.DEVICE.fullmatch(device) for device in args.devices)
    ):
        raise ValueError("Devices must be unique cuda:N identifiers")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    policy_path = args.policy.resolve()
    policy = validate_policy(json.loads(policy_path.read_text()), policy_path)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "policy.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another convergence coordinator holds the policy lock") from error
        state = run_policy(
            policy,
            output,
            args.devices,
            runner=args.runner.resolve(),
            max_attempts=args.max_attempts,
        )
    print(json.dumps(summary(state), indent=2))


if __name__ == "__main__":
    main()
