#!/usr/bin/env python3
"""Report progress of a simple candidate campaign without importing simulation code."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def record_is_retryable(payload, duration_schedule):
    """Return whether an unsettled record still has a declared duration retry."""
    if str(payload.get("status")) != "unsettled" or duration_schedule is None:
        return False
    attempt_index = payload.get("config", {}).get("duration_attempt_index")
    return attempt_index is not None and int(attempt_index) < len(duration_schedule) - 1


def lineage_progress(config, manifest, completed):
    """Return cache progress grouped by declared geometry lineage."""
    manifest = json.loads(Path(manifest).read_text())
    if manifest.get("dataset_id") != config.get("dataset_id"):
        raise ValueError("Campaign dataset_id does not match manifest")
    geometries = {row["geometry_id"]: row for row in manifest["geometries"]}
    conditions = {row["task_id"]: row for row in manifest["conditions"]}
    missing = set(config.get("condition_ids", ())) - conditions.keys()
    if missing:
        raise ValueError(f"Manifest is missing {len(missing)} campaign conditions")

    rows = {}
    candidates = tuple(config.get("candidate_names", ()))
    for condition_id in config.get("condition_ids", ()):
        condition = conditions[condition_id]
        geometry = geometries[condition["geometry_id"]]
        lineage_id = condition["lineage_id"]
        row = rows.setdefault(
            lineage_id,
            dict(
                split=condition["split"],
                epsilon_r=set(),
                planned=0,
                completed=0,
                simulation_status_counts=Counter(),
            ),
        )
        if row["split"] != condition["split"]:
            raise ValueError(f"Lineage {lineage_id} crosses dataset splits")
        row["epsilon_r"].add(geometry["epsilon_r"])
        row["planned"] += len(candidates)
        for candidate in candidates:
            payload = completed.get(f"{condition_id}_{candidate}")
            if payload is not None:
                row["completed"] += 1
                row["simulation_status_counts"][str(payload.get("status", "unknown"))] += 1

    return {
        lineage_id: {
            **row,
            "epsilon_r": sorted(row["epsilon_r"]),
            "remaining": row["planned"] - row["completed"],
            "simulation_status_counts": dict(sorted(row["simulation_status_counts"].items())),
        }
        for lineage_id, row in sorted(rows.items())
    }


def snapshot(campaign, output, window_minutes=10, now=None, manifest=None):
    """Return a tolerant, read-only status snapshot for a campaign cache."""
    campaign = Path(campaign)
    output = Path(output)
    config = json.loads(campaign.read_text())
    expected = {
        f"{condition}_{candidate}"
        for condition in config.get("condition_ids", ())
        for candidate in config.get("candidate_names", ())
    }
    now = time.time() if now is None else now
    observed = {}
    malformed = 0
    mtimes = []
    for case_id in expected:
        path = output / "cases" / case_id / "record.json"
        try:
            mtime = path.stat().st_mtime
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("record is not an object")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            if path.exists():
                malformed += 1
            continue
        observed[case_id] = payload
        mtimes.append(mtime)

    duration_schedule = config.get("duration_schedule_s")
    retrying = {
        case_id: payload
        for case_id, payload in observed.items()
        if record_is_retryable(payload, duration_schedule)
    }
    completed = {
        case_id: payload for case_id, payload in observed.items() if case_id not in retrying
    }
    terminal_mtimes = [
        (output / "cases" / case_id / "record.json").stat().st_mtime
        for case_id in completed
    ]

    window_seconds = max(0.0, window_minutes) * 60
    cutoff = now - window_seconds
    recent = [mtime for mtime in terminal_mtimes if mtime >= cutoff]
    observed_seconds = (
        min(window_seconds, now - min(terminal_mtimes)) if terminal_mtimes else 0.0
    )
    rate = len(recent) * 3600 / observed_seconds if recent and observed_seconds > 0 else None
    remaining = len(expected) - len(completed)
    eta = remaining * 3600 / rate if rate and remaining else (0.0 if not remaining else None)
    outcomes = Counter(str(payload.get("status", "unknown")) for payload in completed.values())
    retrying_outcomes = Counter(
        str(payload.get("status", "unknown")) for payload in retrying.values()
    )
    cache_states = {
        "valid": len(observed),
        "malformed": malformed,
        "missing": len(expected) - len(observed) - malformed,
    }
    result = {
        "campaign": str(campaign),
        "output": str(output),
        "planned": len(expected),
        "completed": len(completed),
        "observed_records": len(observed),
        "remaining": remaining,
        "simulation_status_counts": dict(sorted(outcomes.items())),
        "retrying_status_counts": dict(sorted(retrying_outcomes.items())),
        "cache_status_counts": cache_states,
        "malformed_expected_records": malformed,
        "recent_completions": len(recent),
        "recent_completion_rate_per_hour": rate,
        "estimated_remaining_seconds": eta,
        "earliest_mtime": min(terminal_mtimes) if terminal_mtimes else None,
        "latest_mtime": max(terminal_mtimes) if terminal_mtimes else None,
    }
    if manifest is not None:
        result["lineage_progress"] = lineage_progress(config, manifest, completed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("configs/simple_candidate_full.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/simple_candidate_pilot"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/simple_dielectric_pool.json")
    )
    parser.add_argument("--window-minutes", type=float, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            snapshot(args.campaign, args.output, args.window_minutes, manifest=args.manifest),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
