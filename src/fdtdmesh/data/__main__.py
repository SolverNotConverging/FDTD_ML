"""Generate versioned scenes or evaluate their meshes using real CUDA FDTD."""

import argparse

from .generate import GENERATOR_VERSION, generate_dataset
from .schema import write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", required=True)
    generate.add_argument("--per-split", type=int, default=4)
    generate.add_argument("--seed", type=int, default=2026)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", default="test_iid")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--demo-cnn", action="store_true")
    evaluate.add_argument("--levels", type=int, nargs="+", default=[64, 128, 256, 512])
    evaluate.add_argument("--tolerance", type=float, default=0.02)
    evaluate.add_argument("--consecutive-passes", type=int, default=2)
    evaluate.add_argument("--max-cell-updates", type=int, default=8_000_000_000)
    args = parser.parse_args()
    if args.command == "generate":
        scenes = generate_dataset(args.per_split, args.seed)
        manifest = write_manifest(
            args.output,
            scenes,
            generation={
                "version": GENERATOR_VERSION,
                "seed": args.seed,
                "per_split": args.per_split,
            },
        )
        print(f"Wrote {len(scenes)} scenes; dataset {manifest['dataset_id']}")
    else:
        from fdtdmesh.evaluation import EvaluationConfig, evaluate_dataset

        config = EvaluationConfig(
            reference_levels=tuple(args.levels),
            relative_tolerance=args.tolerance,
            consecutive_passes=args.consecutive_passes,
            max_cell_updates=args.max_cell_updates,
        )
        evaluate_dataset(
            args.manifest,
            args.output,
            config=config,
            split=args.split,
            limit=args.limit,
            checkpoint=args.checkpoint,
            demo_cnn=args.demo_cnn,
        )


if __name__ == "__main__":
    main()
