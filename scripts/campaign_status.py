#!/usr/bin/env python3
"""Read reference campaign progress; no CUDA or project dependencies required."""

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_CAMPAIGN = Path(__file__).resolve().parents[1] / "artifacts/reference_sparse_v7_1000"


def duration(seconds):
    if seconds is None:
        return "not enough data"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"


def snapshot(campaign, now=None):
    campaign = Path(campaign).resolve()
    now = time.time() if now is None else now
    warnings = []

    def read(path):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            warnings.append(f"Could not read {path.name}: {error}")
            return {}

    config = read(campaign / "campaign.json")
    progress = read(campaign / "progress.json")
    stage = read(campaign / "workflow_stage.json")
    complete = read(campaign / "complete.json")
    launch = read(campaign / "workflow_launch.json") or read(campaign / "launch.json")
    targets = config.get("targets", progress.get("targets", {}))
    records = []
    for path in campaign.glob("lane*/decisions/*.json"):
        row = read(path)
        if "status" in row:
            records.append((path.stat().st_mtime, row))
    records.sort(key=lambda item: item[0])
    counts = Counter(row["status"] for _, row in records)
    accepted = counts["converged"]
    total = sum(targets.values())
    by_split = {
        split: {
            "accepted": sum(r["status"] == "converged" and r["split"] == split for _, r in records),
            "target": target,
        }
        for split, target in targets.items()
    }
    started = launch.get("time")
    if started is None and records:
        started = min(t - r.get("wall_seconds", 0) for t, r in records)
    elapsed = max(0, now - started) if started is not None else None
    # Use recent *wall-clock* completion throughput, accounting for all GPUs and
    # rejected attempts. Include time since the last completion to expose stalls.
    window = [(t, r) for t, r in records if started is None or t >= started][-40:]
    rate = None
    if len(window) >= 5:
        interval = now - window[0][0]
        successes = sum(r["status"] == "converged" for _, r in window[1:])
        if interval > 0 and successes:
            rate = successes * 3600 / interval
    remaining = max(0, total - accepted)
    eta = remaining * 3600 / rate if rate else None
    if total and remaining == 0:
        eta = 0
    times = [
        r["wall_seconds"] for _, r in records if r["status"] == "converged" and "wall_seconds" in r
    ]
    resolutions, nt = Counter(), []
    for _, row in records:
        if row["status"] != "converged" or not row.get("reference"):
            continue
        reference = read(campaign / row["reference"])
        budget = reference.get("accepted_budget")
        if budget:
            resolutions["×".join(map(str, budget))] += 1
            levels = reference.get("levels", [])
            selected = next((v for v in reversed(levels) if v.get("budget") == budget), {})
            value = selected.get("diagnostics", {}).get("Nt")
            if value is not None:
                nt.append(value)
    finished_ids = {r["scene_id"] for _, r in records}
    lanes = []
    workers = read(campaign / "workers.json") or []
    for worker in workers:
        lane = campaign / f"lane{worker['lane']}"
        active = read(lane / "active.json")
        scene_id = active.get("scene_id")
        row = dict(lane=worker["lane"], gpu=worker["gpu"], pid=worker["pid"])
        if (lane / "complete.json").exists():
            row["state"] = "quota complete"
        elif not scene_id or scene_id in finished_ids:
            row["state"] = "between cases"
        else:
            detail = read(lane / "attempts" / scene_id / "references" / scene_id / "active.json")
            levels = detail.get("levels", [])
            row.update(
                state="last recorded active case",
                scene_id=scene_id,
                elapsed_seconds=max(0, now - active["started"]),
                event=detail.get("event", "initializing"),
                level=detail.get("level"),
                last_completed_nt=levels[-1].get("diagnostics", {}).get("Nt") if levels else None,
            )
        lanes.append(row)
    failure = read(campaign / "workflow_failure.json")
    if failure and failure.get("time", 0) < stage.get("time", 0):
        failure = {}  # A later workflow stage supersedes a historical failure.
    if failure:
        eta = None
        rate = None
        stage = {"stage": "failed"}
    return dict(
        campaign=str(campaign),
        timestamp=now,
        stage=stage.get("stage", "references_complete" if complete else "sparse_references"),
        recorded_workflow_pid=launch.get("pid"),
        targets=targets,
        accepted=accepted,
        total=total,
        remaining=remaining,
        completed_attempts=len(records),
        counts=dict(counts),
        by_split=by_split,
        elapsed_seconds=elapsed,
        recent_converged_per_hour=rate,
        estimated_remaining_seconds=eta,
        mean_converged_case_seconds=statistics.mean(times) if times else None,
        median_converged_case_seconds=statistics.median(times) if times else None,
        accepted_resolutions=dict(resolutions),
        accepted_nt_range=[min(nt), max(nt)] if nt else None,
        seconds_since_last_decision=max(0, now - records[-1][0]) if records else None,
        lanes=lanes,
        failure=failure,
        warnings=warnings,
    )


def render(data):
    timestamp = (
        datetime.fromtimestamp(data["timestamp"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    total = data["total"]
    percent = 100 * data["accepted"] / total if total else 0
    lines = [
        f"Reference campaign — {timestamp}",
        data["campaign"],
        f"Stage: {data['stage']} | recorded workflow PID: {data['recorded_workflow_pid']}",
        f"Converged: {data['accepted']:,}/{total:,} ({percent:.1f}%) | remaining: {data['remaining']:,}",
        f"Attempts finished: {data['completed_attempts']:,} | outcomes: {data['counts']}",
    ]
    lines.extend(
        f"  {split:12s} {row['accepted']:5,}/{row['target']:,}"
        for split, row in data["by_split"].items()
    )
    lines.extend(
        [
            f"Elapsed: {duration(data['elapsed_seconds'])}",
            f"Mean / median time per converged case: {duration(data['mean_converged_case_seconds'])} / {duration(data['median_converged_case_seconds'])}",
        ]
    )
    if data["recent_converged_per_hour"] is not None:
        lines.append(
            f"Recent throughput: {data['recent_converged_per_hour']:.1f} converged cases/hour across all GPUs"
        )
    if data["failure"]:
        lines.append("Estimated reference time remaining: unavailable while workflow is stopped")
    else:
        lines.append(
            f"Estimated reference time remaining: {duration(data['estimated_remaining_seconds'])} (rough; recent throughput, excludes merge/training)"
        )
    if data["accepted_resolutions"]:
        lines.append(
            "Converged grids: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(data["accepted_resolutions"].items()))
        )
    if data["accepted_nt_range"]:
        a, b = data["accepted_nt_range"]
        lines.append(f"Converged Nt range: {a:,}–{b:,}")
    lines.append(f"Latest finished case: {duration(data['seconds_since_last_decision'])} ago")
    lines.append("GPU lanes (recorded state; this does not check process liveness):")
    for row in data["lanes"]:
        detail = row["state"]
        if "scene_id" in row:
            detail = f"{row['scene_id']} | {row['event']} | grid {row['level'] or '?'} | case elapsed {duration(row['elapsed_seconds'])}"
            if row["last_completed_nt"] is not None:
                detail += f" | last completed Nt {row['last_completed_nt']:,}"
        lines.append(f"  GPU {row['gpu']}: {detail}")
    if data["failure"]:
        lines.append("WORKFLOW FAILURE: " + data["failure"].get("error", str(data["failure"])))
    lines.extend("Warning: " + warning for warning in data["warnings"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="Refresh until Ctrl+C")
    parser.add_argument("--json", action="store_true", help="Machine-readable snapshot")
    args = parser.parse_args()
    if args.watch is not None and (args.watch < 1 or not args.watch < float("inf")):
        parser.error("--watch must be a finite interval of at least 1 second")
    if not (args.campaign / "campaign.json").exists():
        parser.error(f"Campaign not found: {args.campaign}")
    try:
        while True:
            data = snapshot(args.campaign)
            print(json.dumps(data) if args.json else render(data), flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
            if not args.json:
                print("\n" + "─" * 72)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
