"""Seeded multiscale, multi-object scenes with explicit held-out distributions."""

from dataclasses import asdict, dataclass

import numpy as np

from fdtdmesh.constants import C0

from .schema import SCHEMA_VERSION, SPLITS, SceneSpec, geometry_signature

GENERATOR_VERSION = 3


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


@dataclass(frozen=True)
class GenerationConfig:
    min_objects: int = 1
    max_objects: int = 8
    aspect_min: float = 0.3
    aspect_max: float = 3.3
    size_min: float = 0.035
    size_max: float = 0.50
    epsilon_min: float = 1.0
    epsilon_max: float = 30.0
    epsilon_core_max: float = 10.0
    epsilon_core_probability: float = 0.85
    sigma_min: float = 1e-5
    sigma_max: float = 10.0
    lossless_probability: float = 0.2
    pec_probability: float = 0.25
    raster_size: int = 128
    duration_cycles: float = 24.0
    transit_times: float = 4.0

    def __post_init__(self):
        from fdtdmesh.mesh import cell_count

        for name in ("min_objects", "max_objects", "raster_size"):
            object.__setattr__(self, name, cell_count(getattr(self, name)))
        if self.max_objects < self.min_objects or self.max_objects > 32 or self.raster_size < 16:
            raise ValueError("Invalid object count or raster size")
        for low, high in (
            (self.aspect_min, self.aspect_max),
            (self.size_min, self.size_max),
            (self.epsilon_min, self.epsilon_max),
            (self.sigma_min, self.sigma_max),
        ):
            if not np.isfinite([low, high]).all() or not 0 < low <= high:
                raise ValueError("Ranges must be finite, positive and ordered")
        if self.size_min < 4 / self.raster_size or self.size_max > 0.6:
            raise ValueError("Sizes must resolve four raster pixels and fit inside collars")
        if (
            not np.isfinite([self.epsilon_core_max, self.epsilon_core_probability]).all()
            or not self.epsilon_min <= self.epsilon_core_max <= self.epsilon_max
            or not 0 <= self.epsilon_core_probability <= 1
        ):
            raise ValueError("Invalid permittivity mixture cutoff or probability")
        if (
            not np.isfinite(
                [
                    self.lossless_probability,
                    self.pec_probability,
                    self.duration_cycles,
                    self.transit_times,
                ]
            ).all()
            or not 0 <= self.lossless_probability <= 1
            or not 0 <= self.pec_probability <= 1
            or self.duration_cycles < 12
            or self.transit_times <= 0
        ):
            raise ValueError("Invalid material probabilities or simulation duration")

    def to_dict(self):
        return asdict(self)


def sample_permittivity(rng, config, split):
    """Draw ordinary dk from a dominant low-dk component and a smaller high-dk tail."""
    if split == "test_material_ood":
        return log_uniform(rng, config.epsilon_max * 1.2, config.epsilon_max * 4)
    if rng.random() < config.epsilon_core_probability:
        return log_uniform(rng, config.epsilon_min, config.epsilon_core_max)
    return log_uniform(rng, config.epsilon_core_max, config.epsilon_max)


def _bounds(g, domain):
    if g["kind"] == "circle":
        center = np.array(g["center"]) / domain
        radius = g["radius"] / domain
        return center - radius, center + radius
    if g["kind"] == "rectangle":
        return np.array([g["x_position"][0], g["y_position"][0]]) / domain, np.array(
            [g["x_position"][1], g["y_position"][1]]
        ) / domain
    if g["kind"] == "pec_line":
        return np.array([np.min(g["x"]), np.min(g["y"])]) / domain, np.array(
            [np.max(g["x"]), np.max(g["y"])]
        ) / domain
    v = np.array(g["vertices"]) / domain
    return v.min(0), v.max(0)


def _layout(rng, domain, count, split, cfg):
    geometry, materials, boxes = [], [], []
    separation = 4 / cfg.raster_size

    def material():
        if split != "test_material_ood" and rng.random() < cfg.pec_probability:
            return "PEC"
        name = f"object_{len(materials)}"
        sig_range = (
            (cfg.sigma_max * 2, cfg.sigma_max * 20)
            if split == "test_material_ood"
            else (cfg.sigma_min, cfg.sigma_max)
        )
        sigma = (
            0.0
            if split != "test_material_ood" and rng.random() < cfg.lossless_probability
            else log_uniform(rng, *sig_range)
        )
        materials.append(
            dict(name=name, epsilon_r=sample_permittivity(rng, cfg, split), mu_r=1.0, sigma_e=sigma)
        )
        return name

    attempts = 0
    while len(geometry) < count:
        attempts += 1
        if attempts > 3000:
            raise ValueError("Cannot pack requested resolved objects into this domain")
        remaining = count - len(geometry)
        if split == "test_geometry_ood":
            kind = "polygon"
        else:
            kinds = ["rectangle", "quadrilateral", "circle", "triangle"]
            if split != "test_material_ood":
                kinds.append("pec_line")
            if remaining >= 2:
                kinds.extend(["gap", "touching"])
            kind = str(rng.choice(kinds))
        w, h = (
            log_uniform(rng, cfg.size_min, cfg.size_max),
            log_uniform(rng, cfg.size_min, cfg.size_max),
        )
        if kind == "circle":
            radius = log_uniform(
                rng, cfg.size_min * max(domain) / 2, cfg.size_max * min(domain) / 2
            )
            w, h = 2 * radius / domain
        gap = log_uniform(rng, 4 / cfg.raster_size, 0.15) if kind == "gap" else 0.0
        if kind in ("gap", "touching"):
            if 2 * w + gap > 0.68:
                continue
            span = np.array([2 * w + gap, h])
        else:
            span = np.array([w, h])
        if np.any(span > 0.68):
            continue
        center = rng.uniform(0.16 + span / 2, 0.84 - span / 2)
        x, y = center
        new = []

        def rect(x0, x1, y0, y1):
            return dict(
                kind="rectangle",
                material="pending",
                x_position=[x0 * domain[0], x1 * domain[0]],
                y_position=[y0 * domain[1], y1 * domain[1]],
            )

        if kind in ("gap", "touching"):
            new = [
                rect(x - w - gap / 2, x - gap / 2, y - h / 2, y + h / 2),
                rect(x + gap / 2, x + w + gap / 2, y - h / 2, y + h / 2),
            ]
        elif kind == "rectangle":
            new = [rect(x - w / 2, x + w / 2, y - h / 2, y + h / 2)]
        elif kind == "circle":
            new = [
                dict(
                    kind="circle",
                    material="pending",
                    center=(center * domain).tolist(),
                    radius=radius,
                )
            ]
        elif kind == "pec_line":
            # Exactly aligned anchors on a 64-line lattice, with random orientation.
            if rng.random() < 0.5:
                new = [
                    dict(
                        kind="pec_line",
                        x=round(x * 64) / 64 * domain[0],
                        y=[
                            round((y - h / 2) * 64) / 64 * domain[1],
                            round((y + h / 2) * 64) / 64 * domain[1],
                        ],
                    )
                ]
            else:
                new = [
                    dict(
                        kind="pec_line",
                        y=round(y * 64) / 64 * domain[1],
                        x=[
                            round((x - w / 2) * 64) / 64 * domain[0],
                            round((x + w / 2) * 64) / 64 * domain[0],
                        ],
                    )
                ]
        else:
            n = (
                3
                if kind == "triangle"
                else 4
                if kind == "quadrilateral"
                else int(rng.integers(5, 10))
            )
            angle = rng.uniform(0, 2 * np.pi)
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle
            # Affine regular polygons preserve simple topology while varying orientation/aspect.
            points = np.c_[np.cos(theta), np.sin(theta)]
            points = (points - points.min(0)) / np.ptp(points, axis=0) - 0.5
            points = (center + points * [w, h]) * domain
            new = [
                dict(
                    kind="triangle" if n == 3 else "polygon",
                    material="pending",
                    vertices=points.tolist(),
                )
            ]
        bounds = [_bounds(g, domain) for g in new]
        lo = np.min([b[0] for b in bounds], axis=0)
        hi = np.max([b[1] for b in bounds], axis=0)
        if any(
            not np.any(np.maximum(lo - old_hi, old_lo - hi) >= separation)
            for old_lo, old_hi in boxes
        ):
            continue
        for g in new:
            if g["kind"] != "pec_line":
                g["material"] = material()
        geometry.extend(new)
        boxes.append((lo, hi))
    return geometry, materials, boxes


def make_scene(seed, split, index=0, *, config=None):
    cfg = config or GenerationConfig()
    if split not in SPLITS:
        raise ValueError("Unknown split")
    rng = np.random.default_rng(seed)
    lx = (
        log_uniform(rng, 0.012, 0.060)
        if split != "test_scale_ood"
        else log_uniform(rng, 0.07, 0.12)
    )
    domain = np.array([lx, lx * log_uniform(rng, cfg.aspect_min, cfg.aspect_max)])
    f = log_uniform(rng, 8e9, 40e9) if split != "test_scale_ood" else log_uniform(rng, 50e9, 80e9)
    low, high = (
        (cfg.max_objects + 1, cfg.max_objects + 4)
        if split == "test_compositional"
        else (cfg.min_objects, cfg.max_objects)
    )
    count = int(rng.integers(low, high + 1))
    geometry, materials, boxes = _layout(rng, domain, count, split, cfg)
    # Draw separated probes in vacuum, with margin for coarse bilinear stencils.
    probes = []
    for _ in range(4):
        for attempt in range(3000):
            p = rng.uniform(0.19, 0.81, 2)
            if any(np.all(p > lo - 0.04) and np.all(p < hi + 0.04) for lo, hi in boxes):
                continue
            if any(np.linalg.norm(p - q) < 0.08 for q in probes):
                continue
            probes.append(p)
            break
        else:
            raise ValueError("Cannot place separated vacuum probes")
    width = log_uniform(rng, 0.8, 1.5) / f
    source = dict(
        kind="point",
        x=float(probes[0][0] * domain[0]),
        y=float(probes[0][1] * domain[1]),
        normalization="current",
        waveform="gaussian_sine",
        amplitude=0.001,
        frequency=rng.uniform(0.2, 0.65) * f,
        width=width,
        delay=4 * width,
    )
    max_eps = max([1.0] + [m["epsilon_r"] for m in materials])
    duration = max(
        cfg.duration_cycles / f,
        8 * width + cfg.transit_times * np.linalg.norm(domain) * np.sqrt(max_eps) / C0,
    )
    family = (
        "dense_mix"
        if split == "test_compositional"
        else "polygon_ood"
        if split == "test_geometry_ood"
        else "random_mix"
    )
    budgets = [[64, 64], [96, 96]] if split != "test_budget_ood" else [[80, 80], [112, 112]]
    scene = SceneSpec(
        SCHEMA_VERSION,
        f"{split}-{index:05d}",
        f"seed-{seed}",
        seed,
        split,
        family,
        domain.tolist(),
        0.05 * f,
        f,
        duration,
        "float64",
        [cfg.raster_size] * 2,
        budgets,
        dict(
            pml_width=8,
            thickness=(domain / 8).tolist(),
            direction="xy",
            order=3,
            kappa_max=3.0,
            alpha_max=0.05,
            R0=1e-8,
        ),
        materials,
        geometry,
        [source],
        [
            dict(kind="point", x=float(p[0] * domain[0]), y=float(p[1] * domain[1]))
            for p in probes[1:]
        ],
    )
    scene.validate()
    return scene


def generate_dataset(per_split=4, seed=2026, *, config=None):
    from fdtdmesh.mesh import cell_count

    per_split = cell_count(per_split)
    cfg = config or GenerationConfig()
    rng = np.random.default_rng(seed)
    scenes, signatures = [], []
    for split in SPLITS:
        for index in range(per_split):
            for _ in range(1000):
                try:
                    scene = make_scene(int(rng.integers(0, 2**32)), split, index, config=cfg)
                except ValueError:
                    continue  # Packing/resolution failures are resampled, never silently repaired.
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
                raise RuntimeError(
                    "Cannot generate resolved, separated scenes with this configuration"
                )
    return scenes
