"""Portable JSON scene records; all physical coordinates and times are SI."""

import hashlib
import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.mesh import MESH_POLICY, cell_count

SCHEMA_VERSION = 1
SPLITS = (
    "train",
    "validation",
    "test_iid",
    "test_compositional",
    "test_geometry_ood",
    "test_material_ood",
    "test_scale_ood",
    "test_budget_ood",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[3]
    try:

        def git(*args):
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    tree = hashlib.sha256()
    for path in sorted((root / "src" / "fdtdmesh").rglob("*")):
        if path.suffix in (".py", ".pyx", ".cu", ".h"):
            tree.update(path.relative_to(root).as_posix().encode())
            tree.update(path.read_bytes())
    return {
        "git_commit": commit,
        "dirty": dirty,
        "source_sha256": tree.hexdigest(),
        "native_binaries": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src" / "fdtdmesh" / "solver").iterdir())
            if p.suffix in (".pyd", ".so")
        },
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {name: version(name) for name in ("numpy", "scipy", "torch", "fdtdmesh")},
    }


@dataclass
class SceneSpec:
    schema_version: int
    scene_id: str
    group_id: str
    seed: int
    split: str
    family: str
    domain: list
    f_min: float
    f_max: float
    t_end: float
    dtype: str
    raster_shape: list
    budgets: list
    pml: dict
    materials: list
    geometry: list
    sources: list
    receivers: list
    feature_pixels: float = 4.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        scene = cls(**value)
        scene.validate()
        return scene

    @property
    def content_hash(self):
        return digest(self.to_dict())

    def validate(self):
        canonical(self.to_dict())  # Also rejects NaN/Inf and non-JSON data.
        if self.schema_version != SCHEMA_VERSION or self.split not in SPLITS:
            raise ValueError("Unsupported scene schema or split")
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", self.scene_id)
            or not self.group_id
            or isinstance(self.seed, bool)
            or int(self.seed) != self.seed
            or self.seed < 0
        ):
            raise ValueError("Scene identity and nonnegative integer seed are required")
        if len(self.domain) != 2 or len(self.raster_shape) != 2 or min(self.raster_shape) < 4:
            raise ValueError("Expected 2-D domain and raster shape")
        for n in self.raster_shape:
            cell_count(n)
        if not self.budgets or any(len(b) != 2 for b in self.budgets):
            raise ValueError("Expected nonempty x/y cell budgets")
        if self.feature_pixels < 2:
            raise ValueError("At least two pixels per unanchored feature are required")
        if not self.sources or not self.receivers:
            raise ValueError("Dataset scenes require excitation and receivers")
        for source in self.sources:
            if source.get("normalization") != "current" or source.get("kind") != "point":
                raise ValueError("Dataset excitation must be integrated point current")
        for receiver in self.receivers:
            if receiver.get("kind") == "line" and receiver.get("samples") is None:
                raise ValueError("Dataset line receivers require fixed physical samples")
        for budget in self.budgets:
            for n in budget:
                cell_count(n)
            self.build(budget)  # Reuse authoritative material/geometry/probe validation.
        self.check_features()

    def build(self, budget, *, reference=False):
        nx, ny = map(cell_count, budget)
        s = FDTD_2D_Ez(
            *self.domain, nx, ny, self.f_max, f_min=self.f_min, t_end=self.t_end, dtype=self.dtype
        )
        pml = dict(self.pml)
        if reference:
            # Constant physical thickness; refine absorber cells along with the grid.
            counts = []
            for n, length, thickness in zip((nx, ny), self.domain, pml["thickness"]):
                cells = n * thickness / length
                if abs(cells - round(cells)) > 1e-9 or round(cells) < 1:
                    raise ValueError("Uniform reference budget must align PML interfaces")
                counts.append(int(round(cells)))
            pml["pml_width"] = counts
        s.add_PML(**pml)
        for material in self.materials:
            s.add_material(**material)
        for primitive in self.geometry:
            kind = primitive["kind"]
            arguments = {k: v for k, v in primitive.items() if k != "kind"}
            if kind not in ("rectangle", "circle", "triangle", "polygon", "pec_line"):
                raise ValueError(f"Unsupported geometry {kind}")
            getattr(s, "add_" + kind)(**arguments)
        for source in self.sources:
            s.add_source(**source)
        for receiver in self.receivers:
            s.add_receiver(**receiver)
        s.pml.validate_scene(s)
        return s

    def check_features(self):
        # Generated shapes have bounded aspect ratios. Check their bounding spans;
        # circle diameter, polygon/rectangle spans and declared gap width are resolved.
        pixel = np.array(self.domain) / np.array(self.raster_shape[::-1])
        for primitive in self.geometry:
            kind = primitive["kind"]
            if kind == "pec_line":
                continue  # Explicit anchors protect this deliberately zero-width feature.
            if kind == "circle":
                span = np.full(2, 2 * primitive["radius"])
            elif kind == "rectangle":
                span = [np.ptp(primitive["x_position"]), np.ptp(primitive["y_position"])]
            else:
                span = np.ptp(primitive["vertices"], axis=0)
            if np.any(np.asarray(span) / pixel < self.feature_pixels - 1e-9):
                raise ValueError("Unanchored feature is below the raster resolution policy")
        rectangles = [g for g in self.geometry if g["kind"] == "rectangle"]
        for i, a in enumerate(rectangles):
            for b in rectangles[i + 1 :]:
                for axis, other, step in (
                    ("x_position", "y_position", pixel[0]),
                    ("y_position", "x_position", pixel[1]),
                ):
                    overlap = min(a[other][1], b[other][1]) - max(a[other][0], b[other][0])
                    gap = max(a[axis][0], b[axis][0]) - min(a[axis][1], b[axis][1])
                    if overlap > 0 and 1e-12 * step < gap < self.feature_pixels * step:
                        raise ValueError("Unanchored gap is below the raster resolution policy")


def geometry_signature(scene):
    # Normalize lengths via raster sampling; ignore material contrast and physical scale.
    s = scene.build(scene.budgets[0])
    eps, mu, sigma, pec = s.sample(
        (np.arange(64) + 0.5) * s.Lx / 64, (np.arange(64) + 0.5) * s.Ly / 64, raster=True
    )
    return ((eps != 1) | (mu != 1) | (sigma != 0) | pec).ravel()


def validate_splits(scenes):
    ids, groups, signatures = set(), {}, []
    for scene in scenes:
        scene.validate()
        if scene.scene_id in ids:
            raise ValueError("Duplicate scene ID")
        ids.add(scene.scene_id)
        if scene.group_id in groups and groups[scene.group_id] != scene.split:
            raise ValueError("Scene lineage leaks across splits")
        groups[scene.group_id] = scene.split
        mask = geometry_signature(scene)
        for other, other_mask in signatures:
            if other.split != scene.split and np.mean(mask != other_mask) <= 0.005:
                raise ValueError("Near-duplicate normalized geometry leaks across splits")
        signatures.append((scene, mask))


def write_manifest(path, scenes, *, generation):
    validate_splits(scenes)
    records = [scene.to_dict() for scene in scenes]
    value = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": digest(records),
        "generation": generation,
        "mesh_policy": MESH_POLICY,
        "provenance": provenance(),
        "scenes": records,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return value


def read_manifest(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value["schema_version"] != SCHEMA_VERSION or value["mesh_policy"] != MESH_POLICY:
        raise ValueError("Unsupported dataset schema/mesh policy")
    if digest(value["scenes"]) != value["dataset_id"]:
        raise ValueError("Dataset content hash mismatch")
    scenes = [SceneSpec.from_dict(record) for record in value["scenes"]]
    validate_splits(scenes)
    return value, scenes
