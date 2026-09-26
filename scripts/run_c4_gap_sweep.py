#!/usr/bin/env python3
"""Run a bounded C4 dielectric gap sweep and save its physical evidence."""

import argparse
import json
from pathlib import Path

from scattermesh.gap_sweep_v2 import (
    DEFAULT_BUDGETS,
    DEFAULT_GAPS_M,
    DEFAULT_REFERENCE_LEVELS,
    run_c4_gap_sweep,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs_v2" / "c4_gap_sweep_v2_near_field"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gaps-m", type=float, nargs="+", default=DEFAULT_GAPS_M)
    parser.add_argument("--reference-levels", type=int, nargs="+", default=DEFAULT_REFERENCE_LEVELS)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=("uniform", "center", "interface", "wide", "hybrid"),
    )
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("New FDTD campaign runs require the compiled CUDA kernel")
    result = run_c4_gap_sweep(
        args.output.resolve(),
        gaps_m=args.gaps_m,
        reference_levels=args.reference_levels,
        budgets=args.budgets,
        policies=args.policies,
        device=args.device,
    )
    print(json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
