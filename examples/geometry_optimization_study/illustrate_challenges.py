"""Draw conceptual Yee-edge geometry cases for the public-facing report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

OUT = Path("docs/figures/geometry_optimization/conformal_challenges.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def setup(ax, title):
    ax.set(xlim=(0, 4), ylim=(0, 4), aspect="equal", title=title)
    for v in range(5):
        ax.axvline(v, color="#a4b0b8", lw=0.8, zorder=0)
        ax.axhline(v, color="#a4b0b8", lw=0.8, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])


def edge(ax, a, b, point, caption):
    ax.plot((a[0], b[0]), (a[1], b[1]), color="#d44737", lw=4, zorder=3)
    ax.scatter((a[0], b[0]), (a[1], b[1]), color="white", edgecolor="#d44737", s=48, zorder=4)
    ax.annotate(
        caption,
        point,
        xytext=(2, 3.6),
        ha="center",
        va="center",
        fontsize=9,
        color="#8a251f",
        arrowprops={"arrowstyle": "->", "color": "#8a251f"},
    )


fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), constrained_layout=True)

ax = axes[0]
setup(ax, "Thin PEC strip")
ax.add_patch(Rectangle((1.42, 0.4), 0.16, 3.2, facecolor="#33485c", zorder=2))
edge(ax, (1, 2), (2, 2), (1.5, 2), "vacuum endpoints; two PEC crossings")
ax.text(2, -0.55, "A strip can sit between two vacuum Yee nodes.", ha="center", fontsize=10)

ax = axes[1]
setup(ax, "Thin vacuum gap")
ax.add_patch(Rectangle((0.1, 0.4), 1.34, 3.2, facecolor="#33485c", zorder=2))
ax.add_patch(Rectangle((1.58, 0.4), 2.32, 3.2, facecolor="#33485c", zorder=2))
edge(ax, (1, 2), (2, 2), (1.5, 2), "PEC endpoints; hidden air interval")
ax.text(2, -0.55, "A narrow gap needs separate material transitions.", ha="center", fontsize=10)

ax = axes[2]
setup(ax, "Sharp PEC tip")
ax.add_patch(
    Polygon(((0.25, 0.3), (1.52, 2.18), (2.18, 0.3)), closed=True, facecolor="#33485c", zorder=2)
)
edge(ax, (1, 2), (2, 2), (1.52, 2), "small cut segment; donor may be absent")
ax.text(2, -0.55, "A tip can create a short, unsupported conformal edge.", ha="center", fontsize=10)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(OUT)
