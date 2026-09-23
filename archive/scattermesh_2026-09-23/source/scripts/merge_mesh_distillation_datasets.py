#!/usr/bin/env python3
"""Add training-only remediation targets to a frozen distillation dataset."""

import argparse
import json
from pathlib import Path

from scattermesh import merge_distillation_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = merge_distillation_datasets(args.base, args.augmentation, args.output)
    print(
        json.dumps(
            {
                "dataset_id": payload["dataset_id"],
                "example_count": payload["example_count"],
                "split_counts": payload["split_counts"],
                "output": str(args.output / "dataset.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
