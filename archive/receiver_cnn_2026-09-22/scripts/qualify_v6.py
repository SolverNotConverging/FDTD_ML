"""Generate an auditable v6 gallery and optionally profile native reference allocation."""

import argparse
import gc
import json
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.data.campaign import reference_config
from fdtdmesh.data.generate_v6 import RichConfig, feature_audit, make_rich_scene, ownership
from fdtdmesh.data.schema import write_manifest
from fdtdmesh.evaluation.pipeline import run_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = RichConfig()
    scenes = []
    audits = []
    for index in range(args.count):
        scene = make_rich_scene(2026, "validation", index)
        scenes.append(scene)
        audits.append(feature_audit(scene))
        print(
            f"generated {index + 1}/{args.count}: {len(scene.geometry)} objects, {scene.family}",
            flush=True,
        )
    manifest = write_manifest(
        args.output / "manifest.json",
        scenes,
        generation=dict(version=6, config=asdict(cfg), purpose="development_qualification_only"),
    )
    materials = [m for scene in scenes for m in scene.materials]
    summary = dict(
        dataset_id=manifest["dataset_id"],
        scene_count=len(scenes),
        object_counts=[len(s.geometry) for s in scenes],
        ranges={
            key: [min(m[key] for m in materials), max(m[key] for m in materials)]
            for key in ["epsilon_r", "mu_r", "sigma_e", "sigma_h"]
        },
        minimum_visible_core_pixels=min(a["minimum_core_pixels"] for a in audits),
        feature_audits=audits,
    )
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2))
    rows = int(np.ceil(min(len(scenes), 16) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for scene, ax in zip(scenes, axes.ravel()):
        ax.imshow(
            ownership(scene).T,
            origin="lower",
            interpolation="nearest",
            cmap="tab20",
            vmin=0,
            vmax=20,
        )
        ax.set_title(f"{scene.family}: {len(scene.geometry)} objects", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output / "geometry_gallery.png", dpi=160)
    plt.close(fig)
    if args.profile:
        profiles = []
        for level in [1024, 2048, 4096]:
            started = time.perf_counter()
            result = run_scene(
                replace(scenes[0], t_end=1e-15), [level, level], reference_config(), reference=True
            )
            row = dict(
                level=level,
                seconds=time.perf_counter() - started,
                peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                diagnostics=result.diagnostics,
            )
            profiles.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "allocation_profiles.json").write_text(json.dumps(profiles, indent=2))
            del result
            gc.collect()


if __name__ == "__main__":
    main()
