#!/usr/bin/env python3
"""Run the resumable controlled C3 pair qualification campaign."""

import argparse

from scattermesh.c3_qualification_v2 import (
    DEFAULT_BUDGETS,
    DEFAULT_POLICIES,
    DEFAULT_REFERENCE_LEVELS,
    run_c3_pair_qualification,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="runs_v2/c3_pair_qualification_v1",
        help="Campaign directory; cached records are resumed when fingerprints match.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-levels", nargs="+", type=int, default=DEFAULT_REFERENCE_LEVELS)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("New FDTD campaign runs require the compiled CUDA kernel")
    report = run_c3_pair_qualification(
        args.output,
        reference_levels=args.reference_levels,
        budgets=args.budgets,
        policies=args.policies,
        device=args.device,
    )
    print(
        {
            "qualified_scenes": report["qualified_scene_count"],
            "scene_count": report["scene_count"],
            "candidate_cases": report["candidate_case_count"],
            "candidate_status_counts": report["candidate_status_counts"],
        }
    )


if __name__ == "__main__":
    main()
