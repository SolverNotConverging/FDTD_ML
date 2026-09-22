"""Resolved dielectric diversity with anchored rectangular and thin-wire PEC.

The frozen v4/v5 generators remain available. Every v6 scene has its own deterministic
stream, independent of split sizes, worker scheduling, and acceptance decisions.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage

from fdtdmesh.constants import C0, EPS0, MU0
from fdtdmesh.scene import Material

from .generate import _bounds, log_uniform
from .schema import SCHEMA_VERSION, SPLITS, SceneSpec

GENERATOR_VERSION = 6


@dataclass(frozen=True)
class RichConfig:
    min_objects: int = 8
    max_objects: int = 16
    min_span: float = 0.125
    max_span: float = 0.28
    raster_size: int = 128
    visible_core_pixels: float = 3.0
    visible_fraction: float = 0.25
    epsilon_max: float = 30.0
    mu_max: float = 1.0
    magnetic_loss: bool = False
    loss_min: float = 1e-4
    loss_max: float = 1.0
    lossless_probability: float = 0.25
    pec_probability: float = 0.1
    pec_line_probability: float = 0.5
    duration_cycles: float = 48.0
    transit_times: float = 8.0
    max_generation_attempts: int = 1000

    def __post_init__(self):
        if not 4 <= self.min_objects <= self.max_objects <= 32:
            raise ValueError("Require 4–32 objects")
        if not 0.08 <= self.min_span <= self.max_span <= 0.5:
            raise ValueError("Invalid resolved object spans")
        if self.raster_size != 128 or self.visible_core_pixels < 2:
            raise ValueError("v6 requires the 128 raster/probe lattice and resolved visible cores")
        if not 0 < self.visible_fraction <= 1 or not 0 < self.loss_min <= self.loss_max:
            raise ValueError("Invalid visible fraction or loss bounds")
        if self.mu_max != 1 or self.magnetic_loss:
            raise ValueError("v6 requires mu_r=1 and sigma_h=0")
        if self.epsilon_max <= 1 or self.mu_max < 1:
            raise ValueError("Invalid epsilon/permeability ranges")
        if not all(
            0 <= p <= 1
            for p in (self.lossless_probability, self.pec_probability, self.pec_line_probability)
        ):
            raise ValueError("Invalid material probabilities")
        if min(self.duration_cycles, self.transit_times, self.max_generation_attempts) <= 0:
            raise ValueError("Invalid duration/generation limits")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Configuration must be finite")


def ownership(spec, size=None):
    """Render last-writer primitive IDs, including PEC, through authoritative geometry."""
    size = size or spec.raster_shape[0]
    scene = spec.build(spec.budgets[-1])
    scene.primitives = [
        (kind, Material(f"label{i}", epsilon_r=i + 2), data)
        for i, (kind, _, data) in enumerate(scene.primitives)
    ]
    x = (np.arange(size) + 0.5) * scene.Lx / size
    y = (np.arange(size) + 0.5) * scene.Ly / size
    return np.rint(scene.sample(x, y, raster=True)[0] - 1).astype(np.int16)


def feature_audit(spec, config=None):
    """Reject hidden/subpixel components and thin necks after geometric composition."""
    cfg = config or RichConfig()
    labels = ownership(spec)
    rows = []
    for i, primitive in enumerate(spec.geometry):
        mask = labels == i + 1
        if primitive["kind"] == "pec_line":
            lengths = [
                float(np.ptp(np.atleast_1d(primitive[axis])) / spec.domain[j])
                for j, axis in enumerate(("x", "y"))
            ]
            if max(lengths) < cfg.min_span or mask.sum() < 8:
                raise ValueError("PEC line is too short or hidden")
            rows.append(
                dict(
                    primitive=i,
                    kind="anchored_pec_line",
                    visible_pixels=int(mask.sum()),
                    length_fraction=max(lengths),
                    cells_at_1024=[1024 * max(lengths)],
                    component_core_radii_pixels=[],
                )
            )
            continue
        components, count = ndimage.label(mask)
        if count == 0:
            raise ValueError("Fully occluded primitive")
        cores = []
        for component in range(1, count + 1):
            piece = components == component
            radius = float(ndimage.distance_transform_edt(piece).max())
            if radius < cfg.visible_core_pixels or piece.sum() < 24:
                raise ValueError("Visible fragment is below the feature policy")
            opened = ndimage.binary_opening(piece, iterations=2)
            if ndimage.label(opened)[1] != 1:
                raise ValueError("Visible component has an unresolved neck")
            cores.append(radius)
        single = replace(spec, geometry=[primitive])
        original_area = np.count_nonzero(ownership(single))
        fraction = float(mask.sum() / max(original_area, 1))
        if fraction < cfg.visible_fraction:
            raise ValueError("Primitive is mostly hidden")
        lo, hi = _bounds(primitive, np.asarray(spec.domain))
        span = hi - lo
        if np.min(span) < cfg.min_span - 1e-10:
            raise ValueError("Primitive spans fewer than eight cells of a 64-cell grid")
        rows.append(
            dict(
                primitive=i,
                visible_pixels=int(mask.sum()),
                visible_fraction=fraction,
                component_core_radii_pixels=cores,
                span_fraction=span.tolist(),
                cells_at_1024=(1024 * span).tolist(),
            )
        )
    # Bounded vacuum holes must also be resolved. The exterior vacuum is unrestricted.
    holes, n = ndimage.label(labels == 0)
    exterior = set(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]])
    for i in range(1, n + 1):
        if i not in exterior and ndimage.distance_transform_edt(holes == i).max() < 3:
            raise ValueError("Unresolved enclosed vacuum hole")
    return dict(
        objects=rows,
        occupancy_fraction=float(np.mean(labels != 0)),
        minimum_core_pixels=min(
            min(r["component_core_radii_pixels"]) for r in rows if r["component_core_radii_pixels"]
        ),
    )


def _draw(rng, split, index, seed, cfg):
    lx = log_uniform(rng, 0.015, 0.05)
    domain = np.array([lx, lx * log_uniform(rng, 0.65, 1.55)])
    # Vary electrical size independently of physical size; avoid fixed-frequency shortcuts.
    fmax = log_uniform(rng, 0.6, 2.0) * C0 / max(domain)
    carrier = 0.4 * fmax
    count = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
    geometry, materials = [], []
    relation = ("separated", "contact", "overlap", "nested")[index % 4]

    def rectangle(x0, x1, y0, y1):
        return dict(
            kind="rectangle",
            material="pending",
            x_position=[x0 * domain[0], x1 * domain[0]],
            y_position=[y0 * domain[1], y1 * domain[1]],
        )

    # A protected pair supplies controlled topology. Other shapes may intersect each other.
    if relation == "contact":
        geometry = [rectangle(0.18, 0.34, 0.18, 0.42), rectangle(0.34, 0.51, 0.18, 0.42)]
    elif relation == "overlap":
        geometry = [rectangle(0.18, 0.43, 0.18, 0.40), rectangle(0.31, 0.52, 0.28, 0.51)]
    elif relation == "nested":
        geometry = [rectangle(0.18, 0.52, 0.18, 0.52), rectangle(0.27, 0.43, 0.27, 0.43)]
    else:
        geometry = [rectangle(0.18, 0.32, 0.18, 0.40), rectangle(0.37, 0.51, 0.18, 0.40)]
    # Vary pair dimensions/position while preserving exact contact and topology.
    pair_scale = rng.uniform(0.92, 1.04, 2)
    pair_shift = rng.uniform(-0.005, 0.005, 2)
    for g in geometry:
        for axis, name in enumerate(("x_position", "y_position")):
            g[name] = (
                (
                    (np.asarray(g[name]) / domain[axis] - 0.35) * pair_scale[axis]
                    + 0.35
                    + pair_shift[axis]
                )
                * domain[axis]
            ).tolist()
    for _ in range(count - 2):
        w, h = rng.uniform(cfg.min_span, cfg.max_span, 2)
        kind = str(
            rng.choice(["rectangle", "circle", "triangle", "polygon"], p=[0.3, 0.15, 0.25, 0.3])
        )
        if kind == "circle":
            diameter = rng.uniform(cfg.min_span * max(domain), cfg.max_span * min(domain))
            w, h = diameter / domain
        center = rng.uniform(
            np.array([0.16, 0.16]) + [w / 2, h / 2], np.array([0.84, 0.84]) - [w / 2, h / 2]
        )
        axis = int(rng.integers(2))
        span_axis = (w, h)[axis]
        center[axis] = rng.uniform(0.55 + span_axis / 2, 0.84 - span_axis / 2)
        x, y = center
        if kind == "rectangle":
            g = rectangle(x - w / 2, x + w / 2, y - h / 2, y + h / 2)
        elif kind == "circle":
            g = dict(
                kind=kind,
                material="pending",
                center=(center * domain).tolist(),
                radius=diameter / 2,
            )
        else:
            n = 3 if kind == "triangle" else int(rng.integers(4, 8))
            angles = np.arange(n) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi)
            points = np.c_[np.cos(angles), np.sin(angles)]
            points = (points - points.min(0)) / np.ptp(points, axis=0) - 0.5
            g = dict(
                kind=kind,
                material="pending",
                vertices=((center + points * [w, h]) * domain).tolist(),
            )
        lo, hi = _bounds(g, domain)
        # Preserve the designed contact/overlap; keep other shapes above/right of the pair.
        pair_hi = 0.55
        if lo[0] < pair_hi and lo[1] < pair_hi:
            # This whole layout is resampled, never silently shrink an object.
            raise ValueError("Random primitive covers protected interaction")
        geometry.append(g)
    # Global physical quarter-turns/reflections avoid a fixed interaction quadrant.
    # Swapping domain axes on a quarter turn preserves circles as circles.
    for _ in range(int(rng.integers(4))):
        old_y = domain[1]
        for g in geometry:
            if g["kind"] == "rectangle":
                old_x, old_range_y = g["x_position"], g["y_position"]
                g["x_position"] = [old_y - old_range_y[1], old_y - old_range_y[0]]
                g["y_position"] = old_x
            elif g["kind"] == "circle":
                g["center"] = [old_y - g["center"][1], g["center"][0]]
            else:
                g["vertices"] = [[old_y - y, x] for x, y in g["vertices"]]
        domain = domain[::-1].copy()
    if rng.random() < 0.5:
        for g in geometry:
            if g["kind"] == "rectangle":
                g["x_position"] = [domain[0] - g["x_position"][1], domain[0] - g["x_position"][0]]
            elif g["kind"] == "circle":
                g["center"][0] = domain[0] - g["center"][0]
            else:
                g["vertices"] = [[domain[0] - x, y] for x, y in g["vertices"]]
    required_pec = int(rng.integers(2, len(geometry)))
    for i, g in enumerate(geometry):
        if i == required_pec or (i >= 2 and rng.random() < cfg.pec_probability):
            # Keep the designed first dielectric pair intact. PEC bodies are
            # axis-aligned bounding rectangles, quantized BEFORE defining the
            # continuous scene so every reference level retains exact faces.
            lo, hi = _bounds(g, domain)
            lo = np.rint(lo * 64) / 64 * domain
            hi = np.rint(hi * 64) / 64 * domain
            geometry[i] = dict(
                kind="rectangle",
                material="PEC",
                x_position=[float(lo[0]), float(hi[0])],
                y_position=[float(lo[1]), float(hi[1])],
            )
            continue

        # Independent contrast bins; the current campaign fixes permeability to one.
        def contrast(maximum):
            edges = np.geomspace(1, maximum, 5)
            b = int(rng.integers(4))
            return log_uniform(rng, edges[b], edges[b + 1])

        eps = contrast(cfg.epsilon_max)
        mu = 1.0 if cfg.mu_max == 1 else contrast(cfg.mu_max)

        def loss():
            return (
                0.0
                if rng.random() < cfg.lossless_probability
                else log_uniform(rng, cfg.loss_min, cfg.loss_max)
            )

        e_loss, h_loss = loss(), loss() if cfg.magnetic_loss else 0.0
        name = f"object_{i}"
        materials.append(
            dict(
                name=name,
                epsilon_r=eps,
                mu_r=mu,
                sigma_e=e_loss * 2 * np.pi * carrier * EPS0 * eps,
                sigma_h=h_loss * 2 * np.pi * carrier * MU0 * mu,
            )
        )
        g["material"] = name
    # PEC takes precedence: later curved dielectric shapes must not cut holes
    # into rectangular PEC and reintroduce curved/slanted conductor boundaries.
    geometry = [g for g in geometry if g.get("material") != "PEC"] + [
        g for g in geometry if g.get("material") == "PEC"
    ]
    if rng.random() < cfg.pec_line_probability:
        for _ in range(int(rng.integers(1, 3))):
            axis = int(rng.integers(2))
            length = int(rng.integers(8, 21))
            start = int(rng.integers(11, 54 - length))
            fixed = int(rng.integers(11, 54))
            geometry.append(
                dict(
                    kind="pec_line",
                    **{
                        ("x", "y")[axis]: fixed / 64 * domain[axis],
                        ("y", "x")[axis]: [
                            start / 64 * domain[1 - axis],
                            (start + length) / 64 * domain[1 - axis],
                        ],
                    },
                )
            )
    width = 1 / fmax
    duration = max(
        cfg.duration_cycles / fmax,
        8 * width
        + cfg.transit_times
        * np.linalg.norm(domain)
        * np.sqrt(
            max([1] + [m["epsilon_r"] for m in materials])
            * max([1] + [m["mu_r"] for m in materials])
        )
        / C0,
    )
    # Provisional probes in a guaranteed vacuum margin; actual positions are randomized below.
    point = dict(kind="point", x=0.15 * domain[0], y=0.15 * domain[1])
    source = dict(
        **point,
        normalization="current",
        waveform="gaussian_sine",
        amplitude=0.001,
        frequency=carrier,
        width=width,
        delay=4 * width,
    )
    spec = SceneSpec(
        SCHEMA_VERSION,
        f"{split}-v6-{index:06d}",
        f"v6-seed-{seed}",
        seed,
        split,
        relation,
        domain.tolist(),
        0.05 * fmax,
        fmax,
        duration,
        "float64",
        [cfg.raster_size] * 2,
        [[32, 32], [48, 48], [64, 64], [96, 96]],
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
        [point] * 3,
    )
    labels = ownership(spec)
    clear = ndimage.distance_transform_edt(labels == 0) >= 5
    n = cfg.raster_size
    clear[: int(0.16 * n)] = False
    clear[int(0.84 * n) :] = False
    clear[:, : int(0.16 * n)] = False
    clear[:, int(0.84 * n) :] = False
    candidates = np.argwhere(clear)
    rng.shuffle(candidates)
    probes = []
    for pixel in candidates:
        # Nodes of the 128 reference grid, not cell centers; all refinements retain them.
        p = pixel / n
        if all(np.linalg.norm(p - q) >= 0.10 for q in probes):
            probes.append(p)
        if len(probes) == 4:
            break
    if len(probes) != 4:
        raise ValueError("No room for resolved vacuum probes")
    spec.sources[0].update(x=float(probes[0][0] * domain[0]), y=float(probes[0][1] * domain[1]))
    spec.receivers = [
        dict(kind="point", x=float(p[0] * domain[0]), y=float(p[1] * domain[1])) for p in probes[1:]
    ]
    spec.anchor_probes = True
    spec.pec_policy = "rectangles_and_wires"
    spec.validate()
    feature_audit(spec, cfg)
    return spec


def make_rich_scene(seed, split, index, *, config=None):
    cfg = config or RichConfig()
    if split not in SPLITS or seed < 0 or index < 0:
        raise ValueError("Invalid scene stream")
    stream = np.random.SeedSequence([seed, SPLITS.index(split), index, GENERATOR_VERSION])
    rng = np.random.default_rng(stream)
    scene_seed = int(stream.generate_state(1, dtype=np.uint64)[0])
    for attempt in range(cfg.max_generation_attempts):
        try:
            scene = _draw(rng, split, index, scene_seed, cfg)
            scene.generation_attempts = attempt + 1
            return scene
        except ValueError:
            continue
    raise RuntimeError(f"Could not generate a resolved v6 scene: {split}/{index}")
