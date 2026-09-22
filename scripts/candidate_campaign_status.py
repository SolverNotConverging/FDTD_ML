#!/usr/bin/env python3
"""Report progress of a simple candidate campaign without importing simulation code."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def snapshot(campaign, output, window_minutes=10, now=None):
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
    completed = {}
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
        completed[case_id] = payload
        mtimes.append(mtime)

    window_seconds = max(0.0, window_minutes) * 60
    cutoff = now - window_seconds
    recent = [mtime for mtime in mtimes if mtime >= cutoff]
    observed_seconds = min(window_seconds, now - min(mtimes)) if mtimes else 0.0
    rate = len(recent) * 3600 / observed_seconds if recent and observed_seconds > 0 else None
    remaining = len(expected) - len(completed)
    eta = remaining * 3600 / rate if rate and remaining else (0.0 if not remaining else None)
    outcomes = Counter(str(payload.get("status", "unknown")) for payload in completed.values())
    cache_states = {
        "valid": len(completed),
        "malformed": malformed,
        "missing": len(expected) - len(completed) - malformed,
    }
    return {
        "campaign": str(campaign),
        "output": str(output),
        "planned": len(expected),
        "completed": len(completed),
        "remaining": remaining,
        "simulation_status_counts": dict(sorted(outcomes.items())),
        "cache_status_counts": cache_states,
        "malformed_expected_records": malformed,
        "recent_completions": len(recent),
        "recent_completion_rate_per_hour": rate,
        "estimated_remaining_seconds": eta,
        "earliest_mtime": min(mtimes) if mtimes else None,
        "latest_mtime": max(mtimes) if mtimes else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("configs/simple_candidate_full.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/simple_candidate_pilot"))
    parser.add_argument("--window-minutes", type=float, default=10)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.campaign, args.output, args.window_minutes), sort_keys=True))


if __name__ == "__main__":
    main()
