#!/usr/bin/env python3
"""Print the latest atomic learned-mesh training status."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = args.output / "summary.json"
    progress = args.output / "progress.json"
    launch = args.output / "launch.json"
    if summary.exists():
        payload = json.loads(summary.read_text())
    elif progress.exists():
        payload = json.loads(progress.read_text())
    elif launch.exists():
        payload = json.loads(launch.read_text())
    else:
        payload = {"status": "not_started", "output": str(args.output)}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
