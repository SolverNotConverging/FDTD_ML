"""Fixed-window, same-scene comparison against saved point-sampled pilot metrics."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from fdtdmesh.data.campaign import reference_config
from fdtdmesh.data.schema import SceneSpec, provenance
from fdtdmesh.evaluation.pipeline import (
    grids,
    metrics,
    run_scene,
    sample_observables,
    tail_diagnostic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pilot", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--levels", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene_id = f"train-v5-{args.index:06d}"
    manifest_path = args.manifest or (args.pilot / scene_id / "manifest.json")
    spec = SceneSpec.from_dict(json.loads(manifest_path.read_text())["scenes"][0])
    config = replace(
        reference_config(),
        material_averaging="sampled",
        averaging_samples=args.samples,
        averaging_max_samples=args.max_samples,
    )
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                scene=spec.to_dict(),
                config=asdict(config),
                provenance=provenance(),
                started_unix=time.time(),
            ),
            indent=2,
        )
    )
    times, frequencies = grids(spec, config)
    previous = None
    rows = []
    for level in args.levels:
        print("Starting", level, flush=True)
        result = run_scene(spec, [level, level], config, reference=True)
        observations = sample_observables(result, times)
        row = dict(
            level=level,
            tail=tail_diagnostic(spec, observations, config),
            diagnostics=result.diagnostics,
        )
        if previous is not None:
            row["comparison"] = metrics(previous, observations, times, frequencies, config)
        rows.append(row)
        previous = observations
        np.savez_compressed(args.output / f"level_{level}.npz", times=times, waveforms=observations)
        (args.output / "levels.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    (args.output / "complete.json").write_text(json.dumps(dict(finished_unix=time.time())))


if __name__ == "__main__":
    main()
