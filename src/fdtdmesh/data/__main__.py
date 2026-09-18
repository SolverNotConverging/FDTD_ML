"""Generate versioned scenes or evaluate their meshes using real CUDA FDTD."""

import argparse

from .generate import GENERATOR_VERSION, GenerationConfig, generate_dataset, generate_splits
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
    generate.add_argument("--dk-core-max", type=float, default=10.0)
    generate.add_argument("--dk-core-probability", type=float, default=0.85)
    reference_manifest = sub.add_parser(
        "reference-manifest", help="Generate the canonical train/validation/IID reference set"
    )
    reference_manifest.add_argument("--output", required=True)
    reference_manifest.add_argument("--seed", type=int, default=2026)
    reference_manifest.add_argument("--train", type=int, default=64)
    reference_manifest.add_argument("--validation", type=int, default=16)
    reference_manifest.add_argument("--test", type=int, default=32)
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
    references = sub.add_parser(
        "references", help="Generate or resume converged references without candidate runs"
    )
    references.add_argument("--manifest", required=True)
    references.add_argument("--output", required=True)
    references.add_argument("--splits", nargs="+", default=["train", "validation", "test_iid"])
    references.add_argument("--limit", type=int, help="Maximum scenes per selected split")
    references.add_argument("--levels", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    references.add_argument("--tolerance", type=float, default=0.02)
    references.add_argument("--consecutive-passes", type=int, default=2)
    references.add_argument("--max-cell-updates", type=int, default=128_000_000_000)
    references.add_argument("--duration-multiplier", type=float, default=1.0)
    references.add_argument("--duration-extensions", type=int, default=3)
    references.add_argument("--tail-tolerance", type=float, default=0.01)
    references.add_argument(
        "--fixed-window", action="store_true", help="Disable ring-down acceptance gate explicitly"
    )
    args = parser.parse_args()
    if args.command in ("generate", "reference-manifest"):
        if args.command == "reference-manifest":
            config = GenerationConfig()
            counts = {"train": args.train, "validation": args.validation, "test_iid": args.test}
            scenes = generate_splits(counts, args.seed, config=config)
            generation = {
                "version": GENERATOR_VERSION,
                "seed": args.seed,
                "split_counts": counts,
                "purpose": "converged_reference_pilot",
                "config": config.to_dict(),
            }
        else:
            config = GenerationConfig(
                min_objects=args.objects[0],
                max_objects=args.objects[1],
                aspect_min=args.aspect[0],
                aspect_max=args.aspect[1],
                duration_cycles=args.duration_cycles,
                epsilon_core_max=args.dk_core_max,
                epsilon_core_probability=args.dk_core_probability,
            )
            scenes = generate_dataset(args.per_split, args.seed, config=config)
            generation = {
                "version": GENERATOR_VERSION,
                "seed": args.seed,
                "per_split": args.per_split,
                "config": config.to_dict(),
            }
        manifest = write_manifest(
            args.output,
            scenes,
            generation=generation,
        )
        print(f"Wrote {len(scenes)} scenes; dataset {manifest['dataset_id']}")
    elif args.command in ("evaluate", "references"):
        from fdtdmesh.evaluation import EvaluationConfig, evaluate_dataset, generate_references

        config = EvaluationConfig(
            reference_levels=tuple(args.levels),
            relative_tolerance=args.tolerance,
            consecutive_passes=args.consecutive_passes,
            max_cell_updates=args.max_cell_updates,
            duration_multiplier=args.duration_multiplier,
            max_duration_extensions=args.duration_extensions,
            tail_relative_tolerance=None if args.fixed_window else args.tail_tolerance,
        )
        if args.command == "references":
            generate_references(
                args.manifest,
                args.output,
                config=config,
                splits=args.splits,
                limit=args.limit,
            )
        else:
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
