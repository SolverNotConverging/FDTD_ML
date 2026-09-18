"""Generate versioned scenes or evaluate their meshes using real CUDA FDTD."""

import argparse

from .generate import GENERATOR_VERSION, GenerationConfig, generate_dataset
from .schema import write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", required=True)
    generate.add_argument("--per-split", type=int, default=4)
    generate.add_argument("--seed", type=int, default=2026)
    generate.add_argument("--objects", type=int, nargs=2, default=[1, 8], metavar=("MIN", "MAX"))
    generate.add_argument(
        "--aspect", type=float, nargs=2, default=[0.3, 3.3], metavar=("MIN", "MAX")
    )
    generate.add_argument("--duration-cycles", type=float, default=24.0)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", default="test_iid")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--demo-cnn", action="store_true")
    evaluate.add_argument("--levels", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    evaluate.add_argument("--tolerance", type=float, default=0.02)
    evaluate.add_argument("--consecutive-passes", type=int, default=2)
    evaluate.add_argument("--max-cell-updates", type=int, default=128_000_000_000)
    evaluate.add_argument("--duration-multiplier", type=float, default=1.0)
    evaluate.add_argument("--duration-extensions", type=int, default=2)
    evaluate.add_argument("--tail-tolerance", type=float, default=0.01)
    evaluate.add_argument(
        "--fixed-window", action="store_true", help="Disable ring-down acceptance gate explicitly"
    )
    args = parser.parse_args()
    if args.command == "generate":
        config = GenerationConfig(
            min_objects=args.objects[0],
            max_objects=args.objects[1],
            aspect_min=args.aspect[0],
            aspect_max=args.aspect[1],
            duration_cycles=args.duration_cycles,
        )
        scenes = generate_dataset(args.per_split, args.seed, config=config)
        manifest = write_manifest(
            args.output,
            scenes,
            generation={
                "version": GENERATOR_VERSION,
                "seed": args.seed,
                "per_split": args.per_split,
                "config": config.to_dict(),
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
            duration_multiplier=args.duration_multiplier,
            max_duration_extensions=args.duration_extensions,
            tail_relative_tolerance=None if args.fixed_window else args.tail_tolerance,
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
