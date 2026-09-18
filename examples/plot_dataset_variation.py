"""Inspect generated object counts, shape scales, materials and aspect ratios."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from fdtdmesh.data.generate import _bounds
from fdtdmesh.data.schema import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage3_v3"))
    args = parser.parse_args()
    manifest, scenes = read_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    ordinary = [s for s in scenes if s.split in ("train", "validation", "test_iid")]
    selected = [min(ordinary, key=lambda s: abs(len(s.geometry) - n)) for n in range(1, 9)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax, spec in zip(axes.ravel(), selected):
        s = spec.build(spec.budgets[0])
        x = (np.arange(256) + 0.5) * s.Lx / 256
        y = (np.arange(256) + 0.5) * s.Ly / 256
        eps, _, _, pec = s.sample(x, y, raster=True)
        ax.imshow(
            np.ma.masked_where(pec.T, eps.T),
            origin="lower",
            extent=(0, s.Lx * 1e3, 0, s.Ly * 1e3),
            norm=LogNorm(1, 30),
            cmap="YlOrRd",
            aspect="equal",
        )
        ax.contourf(x * 1e3, y * 1e3, pec.T, levels=[0.5, 1.5], colors=["#343c4b"])
        for lo, hi in ((0, s.Lx / 8), (7 * s.Lx / 8, s.Lx)):
            ax.axvspan(lo * 1e3, hi * 1e3, color="gray", alpha=0.2)
        for lo, hi in ((0, s.Ly / 8), (7 * s.Ly / 8, s.Ly)):
            ax.axhspan(lo * 1e3, hi * 1e3, color="gray", alpha=0.2)
        for p in spec.sources:
            ax.plot(p["x"] * 1e3, p["y"] * 1e3, "b*", ms=7)
        for p in spec.receivers:
            ax.plot(p["x"] * 1e3, p["y"] * 1e3, "bo", ms=3)
        ax.set(
            title=f"{len(spec.geometry)} objects / Ly:Lx = {s.Ly / s.Lx:.2f}",
            xlabel="x (mm)",
            ylabel="y (mm)",
        )
    fig.suptitle(
        "Random object counts and multiscale layouts / dark PEC / warm dielectric / blue probes"
    )
    fig.tight_layout()
    fig.savefig(args.output / "diverse_scenes.png", dpi=160)
    plt.close(fig)
    spans = []
    for s in ordinary:
        for g in s.geometry:
            if g["kind"] != "pec_line":
                lo, hi = _bounds(g, np.array(s.domain))
                spans.extend(hi - lo)
    eps = [m["epsilon_r"] for s in ordinary for m in s.materials]
    sig = [m["sigma_e"] for s in ordinary for m in s.materials]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].hist([len(s.geometry) for s in ordinary], bins=np.arange(0.5, 9.5), color="#3274ad")
    axes[0].set_xlabel("Objects per ordinary scene")
    axes[1].hist(
        [s.domain[1] / s.domain[0] for s in ordinary],
        bins=np.geomspace(0.3, 3.3, 12),
        color="#3274ad",
    )
    axes[1].set(xscale="log", xlabel="Aspect ratio Ly / Lx")
    axes[1].set_xticks([0.3, 0.5, 1, 2, 3.3], ["0.3", "0.5", "1", "2", "3.3"])
    axes[1].minorticks_off()
    axes[2].hist(eps, bins=np.geomspace(1, 30, 15), color="#d65f32")
    axes[2].set(xscale="log", xlabel="Per-object relative permittivity")
    axes[3].hist([v for v in sig if v > 0], bins=np.geomspace(1e-5, 10, 16), color="#d65f32")
    axes[3].set(xscale="log", xlabel="Per-object conductivity (S/m)")
    axes[3].set_title(f"Plus {sig.count(0.0)} lossless objects")
    fig.suptitle(f"Measured diversity across {len(ordinary)} training/validation/IID scenes")
    fig.tight_layout()
    fig.savefig(args.output / "diversity_distributions.png", dpi=160)
    plt.close(fig)
    probes = np.array(
        [
            [[p["x"] / s.domain[0], p["y"] / s.domain[1]] for p in s.sources + s.receivers]
            for s in ordinary
        ]
    )
    cutoff = manifest.get("generation", {}).get("config", {}).get("epsilon_core_max", 10.0)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.7))
    names = ["Source", "Receiver 1", "Receiver 2", "Receiver 3"]
    for index, (ax, name) in enumerate(zip(axes, names)):
        ax.axvspan(0, 0.125, color="gray", alpha=0.15)
        ax.axvspan(0.875, 1, color="gray", alpha=0.15)
        ax.axhspan(0, 0.125, color="gray", alpha=0.15)
        ax.axhspan(0.875, 1, color="gray", alpha=0.15)
        ax.scatter(*probes[:, index, :].T, s=15, alpha=0.7, color="#3274ad")
        ax.set(title=name, xlabel="x / Lx", ylabel="y / Ly", xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
    axes[4].hist(eps, bins=np.linspace(1, 30, 16), color="#d65f32")
    axes[4].axvline(cutoff, color="#343c4b", linestyle="--")
    core_fraction = float(np.mean(np.array(eps) <= cutoff))
    axes[4].set(
        title=f"{core_fraction:.1%} of objects: dk <= {cutoff:g}",
        xlabel="Dielectric constant (dk)",
        ylabel="Object count",
    )
    fig.suptitle(
        f"Measured probe positions and dk across {len(ordinary)} train/validation/IID scenes"
    )
    fig.tight_layout()
    fig.savefig(args.output / "probes_and_permittivity.png", dpi=160)
    plt.close(fig)
    summary = {
        "scenes": len(scenes),
        "ordinary_scenes": len(ordinary),
        "object_counts": sorted(set(len(s.geometry) for s in ordinary)),
        "aspect_range": [
            min(s.domain[1] / s.domain[0] for s in ordinary),
            max(s.domain[1] / s.domain[0] for s in ordinary),
        ],
        "normalized_span_range": [min(spans), max(spans)],
        "epsilon_range": [min(eps), max(eps)],
        "epsilon_core_max": cutoff,
        "epsilon_core_fraction": core_fraction,
        "dielectric_objects": len(eps),
        "probe_normalized_ranges": {
            name: {"min": probes[:, i].min(0).tolist(), "max": probes[:, i].max(0).tolist()}
            for i, name in enumerate(names)
        },
        "sigma_range": [min(sig), max(sig)],
        "duration_ns_range": [
            min(s.t_end for s in scenes) * 1e9,
            max(s.t_end for s in scenes) * 1e9,
        ],
    }
    (args.output / "diversity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
