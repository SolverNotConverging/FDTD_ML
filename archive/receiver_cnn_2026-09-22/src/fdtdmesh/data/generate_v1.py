"""Frozen generator v1 for reproducing historical Stage 3 datasets and tests."""

import numpy as np

from .schema import SCHEMA_VERSION, SPLITS, SceneSpec, geometry_signature

GENERATOR_VERSION = 1


def make_scene(seed, split, index=0):
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.018, 0.024) if split != "test_scale_ood" else rng.uniform(0.036, 0.048)
    lx, ly = scale, scale * rng.uniform(0.8, 1.2)
    f = rng.uniform(12e9, 18e9) if split != "test_scale_ood" else rng.uniform(30e9, 40e9)
    eps = rng.uniform(1.5, 4) if split != "test_material_ood" else rng.uniform(8, 12)
    sigma = rng.uniform(0, 0.02) if split != "test_material_ood" else rng.uniform(0.05, 0.15)
    family = ("rectangle", "circle", "thin_line", "gap")[index % 4]
    if split == "test_geometry_ood":
        family = ("triangle", "polygon")[index % 2]
    if split == "test_compositional":
        family = ("mixed", "touching")[index % 2]
    cx, cy = rng.uniform(0.45, 0.58), rng.uniform(0.4, 0.6)
    w, h = rng.uniform(0.10, 0.16), rng.uniform(0.12, 0.22)
    mat = "dielectric" if index % 3 else "PEC"
    if split == "test_material_ood":
        family = ("rectangle", "circle")[index % 2]
        mat = "dielectric"

    def rectangle(x0, x1, y0, y1, material=mat):
        return dict(
            kind="rectangle",
            material=material,
            x_position=[x0 * lx, x1 * lx],
            y_position=[y0 * ly, y1 * ly],
        )

    def circle(x, y, r, material=mat):
        return dict(
            kind="circle", material=material, center=[x * lx, y * ly], radius=r * min(lx, ly)
        )

    if family == "rectangle":
        geometry = [rectangle(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)]
    elif family == "circle":
        geometry = [circle(cx, cy, w / 2)]
    elif family == "thin_line":
        # Dyadic anchors are exactly representable by 64*2**k uniform references.
        geometry = [
            dict(
                kind="pec_line",
                x=round(cx * 32) / 32 * lx,
                y=[round((cy - h / 2) * 32) / 32 * ly, round((cy + h / 2) * 32) / 32 * ly],
            )
        ]
    elif family == "gap":
        gap = rng.uniform(0.07, 0.09)  # >=4 pixels at 64x64.
        geometry = [
            rectangle(cx - w - gap / 2, cx - gap / 2, cy - h / 2, cy + h / 2),
            rectangle(cx + gap / 2, cx + w + gap / 2, cy - h / 2, cy + h / 2),
        ]
    elif family == "touching":
        geometry = [
            rectangle(cx - w, cx, cy - h / 2, cy + h / 2, "PEC"),
            rectangle(cx, cx + w, cy - h / 2, cy + h / 2, "dielectric"),
        ]
    elif family == "mixed":
        geometry = [
            rectangle(cx - 0.16, cx - 0.05, cy - 0.16, cy + 0.02, "PEC"),
            circle(cx + 0.12, cy + 0.08, w / 2, "dielectric"),
        ]
    else:
        vertices = (
            [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx, cy + h / 2)]
            if family == "triangle"
            else [
                (cx - w / 2, cy - h / 4),
                (cx, cy - h / 2),
                (cx + w / 2, cy - h / 4),
                (cx + w / 3, cy + h / 2),
                (cx - w / 3, cy + h / 2),
            ]
        )
        geometry = [
            dict(kind=family, material=mat, vertices=[[x * lx, y * ly] for x, y in vertices])
        ]
    budgets = [[64, 64], [96, 96]] if split != "test_budget_ood" else [[80, 80], [112, 112]]
    scene = SceneSpec(
        SCHEMA_VERSION,
        f"{split}-{index:05d}",
        f"seed-{seed}",
        seed,
        split,
        family,
        [lx, ly],
        0.1 * f,
        f,
        12 / f,
        "float64",
        [64, 64],
        budgets,
        dict(
            pml_width=8,
            thickness=[lx / 8, ly / 8],
            direction="xy",
            order=3,
            kappa_max=3.0,
            alpha_max=0.05,
            R0=1e-8,
        ),
        [dict(name="dielectric", epsilon_r=eps, mu_r=1.0, sigma_e=sigma)],
        geometry,
        [
            dict(
                kind="point",
                x=0.23 * lx,
                y=rng.uniform(0.40, 0.60) * ly,
                normalization="current",
                waveform="gaussian_sine",
                amplitude=0.001,
                frequency=0.4 * f,
                width=1 / f,
                delay=4 / f,
            )
        ],
        [dict(kind="point", x=0.77 * lx, y=y * ly) for y in (0.35, 0.5, 0.65)],
    )
    scene.validate()
    return scene


def generate_dataset(per_split=4, seed=2026):
    if isinstance(per_split, bool) or int(per_split) != per_split or per_split < 1:
        raise ValueError("per_split must be a positive integer")
    rng = np.random.default_rng(seed)
    scenes, signatures = [], []
    for split in SPLITS:
        for index in range(per_split):
            for _ in range(1000):
                scene = make_scene(int(rng.integers(0, 2**32)), split, index)
                mask = geometry_signature(scene)
                if any(
                    other.split != split and np.mean(mask != signature) <= 0.005
                    for other, signature in zip(scenes, signatures)
                ):
                    continue
                scenes.append(scene)
                signatures.append(mask)
                break
            else:
                raise RuntimeError("Cannot construct separated splits; reduce requested count")
    return scenes
