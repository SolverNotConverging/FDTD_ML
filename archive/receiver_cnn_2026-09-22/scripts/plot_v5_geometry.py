"""Plot physical geometry and actual reference mesh anchors from a scene manifest."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.data.schema import SceneSpec


def draw(ax, spec, *, grid=False, anchors=False):
    scene = spec.build([128, 128], reference=True)
    mesh = scene.mesh_uniform()
    n = 512
    x = (np.arange(n) + 0.5) * scene.Lx / n
    y = (np.arange(n) + 0.5) * scene.Ly / n
    eps, _, _, pec = scene.sample(x, y)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    im = ax.imshow(
        np.ma.masked_where(pec.T, eps.T),
        origin="lower",
        extent=[0, scene.Lx * 1000, 0, scene.Ly * 1000],
        cmap=cmap,
        vmin=1,
        vmax=30,
        interpolation="nearest",
    )
    if grid:
        ax.vlines(mesh.x * 1000, 0, scene.Ly * 1000, color="white", lw=0.25, alpha=0.5)
        ax.hlines(mesh.y * 1000, 0, scene.Lx * 1000, color="white", lw=0.25, alpha=0.5)
    if anchors:
        for v in scene.x_anchors:
            ax.axvline(v * 1000, color="#ff9533", ls="--", lw=0.8)
        for v in scene.y_anchors:
            ax.axhline(v * 1000, color="#ff9533", ls="--", lw=0.8)
    for g in spec.geometry:
        if g["kind"] == "pec_line":
            xx, yy = np.broadcast_arrays(np.atleast_1d(g["x"]), np.atleast_1d(g["y"]))
            ax.plot(xx * 1000, yy * 1000, "k-", lw=2.2)
            ax.plot(xx * 1000, yy * 1000, "ks", ms=3)
    for p in spec.sources:
        ax.plot(p["x"] * 1000, p["y"] * 1000, "*", color="#ff4242", mec="white", mew=0.4, ms=10)
    for p in spec.receivers:
        ax.plot(p["x"] * 1000, p["y"] * 1000, "x", color="#37cfff", mew=1.7, ms=6)
    ax.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal")
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    specs = [SceneSpec.from_dict(s) for s in json.loads(args.manifest.read_text())["scenes"]]
    specs = specs[:args.count]
    if not specs:
        parser.error("At least one scene is required")
    root = args.manifest.parent
    fig, axes = plt.subplots((len(specs) + 1) // 2, 2, figsize=(13, 6 * ((len(specs) + 1) // 2)), layout="constrained", squeeze=False)
    for ax, spec in zip(axes.flat, specs):
        im = draw(ax, spec)
        lines = sum(g["kind"] == "pec_line" for g in spec.geometry)
        ax.set_title(
            f"{spec.scene_id.rsplit('-', 1)[-1]} · {spec.family}\n{len(spec.geometry) - lines} bodies, {lines} PEC lines",
            fontsize=10,
        )
    for ax in list(axes.flat)[len(specs):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=list(axes.flat), shrink=0.7, label="Relative permittivity εr (black = PEC)")
    fig.suptitle(
        "New geometry samples: μr = 1, σh = 0\nRed star: source/port · cyan crosses: receivers · black: PEC bodies and lines",
        fontsize=14,
    )
    fig.savefig(root / "pec_geometry_gallery.png", dpi=160)
    plt.close(fig)
    spec = next((s for s in specs if any(g["kind"] == "pec_line" for g in s.geometry)), None)
    if spec is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for ax in axes:
        draw(ax, spec, grid=True, anchors=True)
    line = next(g for g in spec.geometry if g["kind"] == "pec_line")
    cx, cy = np.mean(line["x"]) * 1000, np.mean(line["y"]) * 1000
    dx, dy = spec.domain[0] * 1000 * 0.19, spec.domain[1] * 1000 * 0.19
    axes[1].set_xlim(max(0, cx - dx), min(spec.domain[0] * 1000, cx + dx))
    axes[1].set_ylim(max(0, cy - dy), min(spec.domain[1] * 1000, cy + dy))
    axes[0].set_title("Full domain: actual 128 × 128 starting reference mesh")
    axes[1].set_title("PEC-line detail: endpoints and fixed coordinate anchored")
    fig.suptitle(
        f"{spec.scene_id}: {spec.family}\nWhite: mesh lines · orange dashed: mandatory anchors · black squares: PEC-line endpoints",
        fontsize=13,
    )
    fig.savefig(root / "pec_anchor_mesh.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
