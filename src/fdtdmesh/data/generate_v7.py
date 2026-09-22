"""Sparse one/two-object supplement; the v6 stream remains frozen."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from fdtdmesh.constants import C0, EPS0

from .generate import _bounds, log_uniform
from .schema import SCHEMA_VERSION, SPLITS, SceneSpec

GENERATOR_VERSION = 7
FAMILIES = ("single_dielectric", "dielectric_gap", "single_pec", "pec_dielectric")
SCALE_BANDS = ((0.08, 0.125), (0.125, 0.22), (0.22, 0.30))


@dataclass(frozen=True)
class SparseConfig:
    raster_size: int = 128
    gap_pixels_min: int = 4
    gap_pixels_max: int = 8
    epsilon_max: float = 30.0
    loss_min: float = 1e-4
    loss_max: float = 1.0
    lossless_probability: float = 0.25
    duration_cycles: float = 48.0
    transit_times: float = 8.0
    max_generation_attempts: int = 1000

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Configuration must be finite")
        if self.raster_size != 128:
            raise ValueError("Sparse scenes use the current 128-pixel input")
        if not 4 <= self.gap_pixels_min <= self.gap_pixels_max <= 12:
            raise ValueError("Gaps must span 4–12 input pixels")
        if not 1 < self.epsilon_max <= 30 or not 0 < self.loss_min <= self.loss_max:
            raise ValueError("Invalid material range")
        if not 0 <= self.lossless_probability <= 1:
            raise ValueError("Invalid lossless probability")
        if min(self.duration_cycles, self.transit_times, self.max_generation_attempts) <= 0:
            raise ValueError("Invalid generation limits")


def feature_audit(spec, config=None):
    """Foreground resolution and gap checks independent of empty-domain area."""
    cfg = config or SparseConfig()
    sim = spec.build([128, 128])
    n = cfg.raster_size
    x, y = [(np.arange(n) + 0.5) * d / n for d in spec.domain]
    eps, _, _, pec = sim.sample(x, y, raster=True)
    bounds, objects = [], []
    for g in spec.geometry:
        lo, hi = _bounds(g, np.asarray(spec.domain))
        span = hi - lo
        if min(span) < SCALE_BANDS[0][0] - 1e-10:
            raise ValueError("Object is below the sparse feature-size floor")
        if np.any(lo < 0.16) or np.any(hi > 0.84):
            raise ValueError("Object enters the PML/probe margin")
        bounds.append((lo, hi))
        objects.append(
            dict(kind=g["kind"], span_fraction=span.tolist(), cells_at_1024=(span * 1024).tolist())
        )
    gap = None
    if len(bounds) == 2:
        gap = float(
            np.max(np.maximum(bounds[0][0], bounds[1][0]) - np.minimum(bounds[0][1], bounds[1][1]))
        )
        if gap * n < cfg.gap_pixels_min - 1e-9:
            raise ValueError("Pair gap is unresolved")
    return dict(
        objects=objects,
        occupancy_fraction=float(np.mean((eps != 1) | pec)),
        gap_pixels=None if gap is None else gap * n,
        gap_cells_at_1024=None if gap is None else gap * 1024,
    )


def _draw(rng, split, index, seed, cfg):
    lx = log_uniform(rng, 0.015, 0.05)
    domain = np.array([lx, lx * log_uniform(rng, 0.75, 1.33)])
    fmax = log_uniform(rng, 0.6, 2.0) * C0 / max(domain)
    family = FAMILIES[index % 4]
    band = SCALE_BANDS[(index // 4) % len(SCALE_BANDS)]
    kinds = ("circle", "rectangle", "triangle", "polygon")
    has_pec = family in ("single_pec", "pec_dielectric")
    count = 2 if family in ("dielectric_gap", "pec_dielectric") else 1
    axis = int(rng.integers(2))
    # Layout on a 64-line lattice keeps the PEC continuous geometry identical
    # across reference refinements. Dielectric curves themselves are continuous.
    sizes = []
    selected = []
    for i in range(count):
        offset = i if (index // 48) % 2 else 0
        kind = "rectangle" if has_pec and i == 0 else kinds[(index // 12 + offset) % 4]
        span = np.ceil(rng.uniform(*band, 2) * 64) / 64
        if kind == "circle":
            diameter = max(span * domain)
            span = diameter / domain
        sizes.append(span)
        selected.append(kind)
    gap = rng.integers(int(np.ceil(cfg.gap_pixels_min / 2)), cfg.gap_pixels_max // 2 + 1) / 64
    starts = [np.zeros(2)]
    if count == 2:
        offset = np.zeros(2)
        offset[axis] = sizes[0][axis] + gap
        offset[1 - axis] = (sizes[0][1 - axis] - sizes[1][1 - axis]) / 2
        starts.append(offset)
    lo = np.min(starts, axis=0)
    hi = np.max([p + s for p, s in zip(starts, sizes)], axis=0)
    lower = np.ceil((0.17 - lo) * 64).astype(int)
    upper = np.floor((0.83 - hi) * 64).astype(int)
    if np.any(lower > upper):
        raise ValueError("Cluster is too large for physical interior")
    shift = rng.integers(lower, upper + 1) / 64
    geometry, materials = [], []
    for i, (kind, size, start) in enumerate(zip(selected, sizes, starts)):
        a = start + shift
        b = a + size
        is_pec = has_pec and i == 0
        material = "PEC" if is_pec else f"object_{i}"
        if not is_pec:
            eps = log_uniform(rng, 1.05, cfg.epsilon_max)
            loss = (
                0.0
                if rng.random() < cfg.lossless_probability
                else log_uniform(rng, cfg.loss_min, cfg.loss_max)
            )
            materials.append(
                dict(
                    name=material,
                    epsilon_r=eps,
                    mu_r=1.0,
                    sigma_e=loss * 2 * np.pi * 0.4 * fmax * EPS0 * eps,
                    sigma_h=0.0,
                )
            )
        if kind == "rectangle":
            g = dict(
                kind=kind,
                material=material,
                x_position=[a[0] * domain[0], b[0] * domain[0]],
                y_position=[a[1] * domain[1], b[1] * domain[1]],
            )
        elif kind == "circle":
            g = dict(
                kind=kind,
                material=material,
                center=((a + b) / 2 * domain).tolist(),
                radius=size[0] * domain[0] / 2,
            )
        else:
            n = 3 if kind == "triangle" else int(rng.integers(4, 8))
            angles = np.arange(n) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi)
            points = np.c_[np.cos(angles), np.sin(angles)]
            points = (points - points.min(0)) / np.ptp(points, axis=0)
            g = dict(kind=kind, material=material, vertices=((a + points * size) * domain).tolist())
        geometry.append(g)
    for axis in range(2):
        if rng.random() < 0.5:
            for g in geometry:
                if g["kind"] == "rectangle":
                    key = ("x_position", "y_position")[axis]
                    g[key] = [domain[axis] - v for v in g[key][::-1]]
                elif g["kind"] == "circle":
                    g["center"][axis] = domain[axis] - g["center"][axis]
                else:
                    for p in g["vertices"]:
                        p[axis] = domain[axis] - p[axis]
    geometry.sort(key=lambda g: g["material"] == "PEC")
    width = 1 / fmax
    duration = max(
        cfg.duration_cycles / fmax,
        8 * width
        + cfg.transit_times
        * np.linalg.norm(domain)
        * np.sqrt(max([1] + [m["epsilon_r"] for m in materials]))
        / C0,
    )
    point = dict(kind="point", x=0.15 * domain[0], y=0.15 * domain[1])
    spec = SceneSpec(
        schema_version=SCHEMA_VERSION,
        scene_id=f"{split}-v7-{index:06d}",
        group_id=f"v7-seed-{seed}",
        seed=seed,
        split=split,
        family=family,
        domain=domain.tolist(),
        f_min=0.05 * fmax,
        f_max=fmax,
        t_end=duration,
        dtype="float64",
        raster_shape=[128, 128],
        budgets=[[n, n] for n in (48, 64, 96, 128)],
        pml=dict(
            pml_width=8,
            thickness=(domain / 8).tolist(),
            direction="xy",
            order=3,
            kappa_max=3.0,
            alpha_max=0.05,
            R0=1e-8,
        ),
        materials=materials,
        geometry=geometry,
        sources=[
            dict(
                **point,
                normalization="current",
                waveform="gaussian_sine",
                amplitude=0.001,
                frequency=0.4 * fmax,
                width=width,
                delay=4 * width,
            )
        ],
        receivers=[point],
        anchor_probes=True,
        pec_policy="rectangles_and_wires",
    )
    sim = spec.build([128, 128])
    xy = [np.arange(128) * d / 128 for d in domain]
    eps, _, _, pec = sim.sample(*xy)
    clear = ndimage.distance_transform_edt((eps == 1) & ~pec) >= 5
    clear[:22] = clear[107:] = False
    clear[:, :22] = clear[:, 107:] = False
    candidates = np.argwhere(clear)
    rng.shuffle(candidates)
    probes = []
    for pixel in candidates:
        p = pixel / 128
        if all(np.linalg.norm(p - q) >= 0.1 for q in probes):
            probes.append(p)
        if len(probes) == 4:
            break
    if len(probes) != 4:
        raise ValueError("Insufficient vacuum probe clearance")
    spec.sources[0].update(x=float(probes[0][0] * domain[0]), y=float(probes[0][1] * domain[1]))
    spec.receivers = [
        dict(kind="point", x=float(p[0] * domain[0]), y=float(p[1] * domain[1])) for p in probes[1:]
    ]
    spec.validate()
    feature_audit(spec, cfg)
    return spec


def make_sparse_scene(seed, split, index, *, config=None):
    cfg = config or SparseConfig()
    if split not in SPLITS or seed < 0 or index < 0:
        raise ValueError("Invalid scene stream")
    stream = np.random.SeedSequence([seed, SPLITS.index(split), index, GENERATOR_VERSION])
    rng = np.random.default_rng(stream)
    scene_seed = int(stream.generate_state(1, dtype=np.uint64)[0])
    for attempt in range(cfg.max_generation_attempts):
        try:
            spec = _draw(rng, split, index, scene_seed, cfg)
            spec.generation_attempts = attempt + 1
            return spec
        except ValueError:
            continue
    raise RuntimeError(f"Could not generate resolved sparse scene: {split}/{index}")
