#!/usr/bin/env python3
"""Build validated set-valued mesh-density targets from a completed campaign."""

import argparse
import json
from pathlib import Path

from scattermesh.distillation import build_distillation_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/simple_dielectric_factorial_pool.json")
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/simple_factorial_exact_candidate_full.json")
    )
    parser.add_argument("--campaign-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-bins", type=int, default=128)
    args = parser.parse_args()
    payload = build_distillation_dataset(
        args.manifest,
        args.campaign,
        args.campaign_output,
        args.output,
        profile_bins=args.profile_bins,
    )
    print(
        json.dumps(
            {
                "dataset_id": payload["dataset_id"],
                "campaign_id": payload["campaign_id"],
                "example_count": payload["example_count"],
                "split_counts": payload["split_counts"],
                "profile_bins": payload["profile_bins"],
                "output": str(args.output / "dataset.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
