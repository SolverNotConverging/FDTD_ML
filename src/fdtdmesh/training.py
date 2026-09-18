"""Stage-4 heuristic-teacher targets, reproducible imitation training and evaluation."""

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fdtdmesh.data.schema import provenance, read_manifest
from fdtdmesh.evaluation.metrics import compare_observables, sample_observables
from fdtdmesh.evaluation.pipeline import EvaluationConfig, grids, heuristic_density, run_scene
from fdtdmesh.mesh import MESH_POLICY, projected_density
from fdtdmesh.ml import (
    ResUNet,
    conditioning,
    load_model,
    pool_axes,
    rasterize,
    repair_loss,
    save_model,
)

TEACHER_VERSION = 1
TRAINING_VERSION = 1


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_state():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", None


def build_teacher_targets(
    manifest_path, output_path, *, splits=("train", "validation"), time_limit=30.0
):
    """Project heuristic densities to legal meshes and save their raster-bin mass targets."""
    manifest, scenes = read_manifest(manifest_path)
    selected = [scene for scene in scenes if scene.split in set(splits)]
    if not selected:
        raise ValueError("No scenes selected for teacher targets")
    shapes = {tuple(scene.raster_shape) for scene in selected}
    if len(shapes) != 1:
        raise ValueError("Teacher target file requires one common raster shape")
    shape = shapes.pop()
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() or metadata_path.exists():
        raise ValueError("Teacher output already exists; choose a new path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples, target_x, target_y = [], [], []
    started = perf_counter()
    for scene in selected:
        for budget in scene.budgets:
            simulation = scene.build(budget)
            density = heuristic_density(simulation, shape)
            mesh = simulation.mesh_from_density(*density, time_limit=time_limit)
            target_x.append(projected_density(mesh.x, shape[1], collar=simulation.pml.x))
            target_y.append(projected_density(mesh.y, shape[0], collar=simulation.pml.y))
            samples.append(
                {
                    "scene_id": scene.scene_id,
                    "scene_hash": scene.content_hash,
                    "split": scene.split,
                    "budget": list(budget),
                    "teacher_projection": {
                        key: value
                        for key, value in mesh.metadata.items()
                        if key.endswith(("projection_l1", "projection_max", "projection_status"))
                    },
                }
            )
    np.savez_compressed(
        output_path,
        target_x=np.asarray(target_x, dtype=np.float32),
        target_y=np.asarray(target_y, dtype=np.float32),
    )
    metadata = {
        "schema_version": 1,
        "teacher_version": TEACHER_VERSION,
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": _sha256(manifest_path),
        "targets_sha256": _sha256(output_path),
        "mesh_policy": MESH_POLICY,
        "raster_shape": list(shape),
        "splits": list(splits),
        "teacher": "material_edge_heuristic_projected_to_legal_mesh_then_rebinned",
        "samples": samples,
        "seconds": perf_counter() - started,
        "provenance": provenance(),
    }
    _json(metadata_path, metadata)
    return metadata


def load_teacher_targets(manifest_path, target_path):
    manifest, scenes = read_manifest(manifest_path)
    target_path = Path(target_path)
    metadata = json.loads(target_path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("teacher_version") != TEACHER_VERSION
        or metadata.get("dataset_id") != manifest["dataset_id"]
        or metadata.get("mesh_policy") != MESH_POLICY
        or metadata.get("targets_sha256") != _sha256(target_path)
    ):
        raise ValueError("Teacher targets do not match the dataset or mesh policy")
    arrays = np.load(target_path)
    target_x, target_y = arrays["target_x"], arrays["target_y"]
    if len(target_x) != len(metadata["samples"]) or len(target_y) != len(metadata["samples"]):
        raise ValueError("Teacher target count mismatch")
    scene_map = {scene.scene_id: scene for scene in scenes}
    for sample in metadata["samples"]:
        if (
            sample["scene_id"] not in scene_map
            or scene_map[sample["scene_id"]].content_hash != sample["scene_hash"]
        ):
            raise ValueError("Teacher sample scene mismatch")
    return manifest, scene_map, metadata, target_x, target_y


class TeacherDataset(Dataset):
    def __init__(self, manifest_path, target_path, split):
        manifest, scene_map, metadata, target_x, target_y = load_teacher_targets(
            manifest_path, target_path
        )
        self.dataset_id = manifest["dataset_id"]
        self.target_id = metadata["targets_sha256"]
        self.raster_shape = tuple(metadata["raster_shape"])
        self.records = []
        for index, sample in enumerate(metadata["samples"]):
            if sample["split"] != split:
                continue
            scene, budget = scene_map[sample["scene_id"]], sample["budget"]
            simulation = scene.build(budget)
            self.records.append(
                {
                    "scene": scene,
                    "budget": budget,
                    "raster": torch.from_numpy(
                        rasterize(simulation, self.raster_shape, scene.f_max)
                    ),
                    "condition": torch.from_numpy(
                        conditioning(simulation, *budget, scene.f_max, scene.f_min)
                    ),
                    "target_x": torch.from_numpy(target_x[index].copy()),
                    "target_y": torch.from_numpy(target_y[index].copy()),
                    "collar": torch.tensor(
                        [
                            simulation.pml.x.thickness / simulation.Lx,
                            simulation.pml.y.thickness / simulation.Ly,
                        ],
                        dtype=torch.float32,
                    ),
                }
            )
        if not self.records:
            raise ValueError(f"No {split} teacher samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {key: value for key, value in record.items() if key not in ("scene", "budget")} | {
            "index": index
        }


def axis_probability(density, collar_fraction):
    if density.ndim != 2 or collar_fraction.shape != (len(density),):
        raise ValueError("Expected batched density and one normalized collar per sample")
    edges = torch.linspace(0, 1, density.shape[1] + 1, device=density.device, dtype=density.dtype)
    widths = torch.clamp(
        torch.minimum(edges[1:, None], 1 - collar_fraction[None, :])
        - torch.maximum(edges[:-1, None], collar_fraction[None, :]),
        min=0,
    ).T
    mass = density * widths
    return mass / mass.sum(1, keepdim=True)


def imitation_loss(rho_x, rho_y, target_x, target_y, collars):
    px = axis_probability(rho_x, collars[:, 0])
    py = axis_probability(rho_y, collars[:, 1])
    lx = (px.cumsum(1) - target_x.cumsum(1)).square().mean()
    ly = (py.cumsum(1) - target_y.cumsum(1)).square().mean()
    return (lx + ly) / 2, lx, ly


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    width: int = 16
    alpha: float = 4.0
    repair_weight: float = 0.05
    repair_every: int = 4
    projection_samples: int = 8
    seed: int = 2026

    def __post_init__(self):
        integer = ("epochs", "batch_size", "width", "repair_every", "projection_samples", "seed")
        for name in integer:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or int(value) != value
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError(f"Invalid training {name}")
        values = [self.learning_rate, self.weight_decay, self.alpha, self.repair_weight]
        if (
            not np.isfinite(values).all()
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or self.alpha <= 0
            or self.repair_weight < 0
        ):
            raise ValueError("Invalid training scalar")


def _project_batch(dataset, indices, rho_x, rho_y, time_limit=30.0):
    meshes, x_collars, y_collars = [], [], []
    for offset, index in enumerate(indices.tolist()):
        record = dataset.records[index]
        simulation = record["scene"].build(record["budget"])
        mesh = simulation.mesh_from_density(
            rho_x[offset].detach().cpu().numpy(),
            rho_y[offset].detach().cpu().numpy(),
            time_limit=time_limit,
        )
        meshes.append(mesh)
        x_collars.append(simulation.pml.x)
        y_collars.append(simulation.pml.y)
    return meshes, x_collars, y_collars


def evaluate_imitation(model, dataset, config, device, *, project=True):
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    totals = {"loss": 0.0, "loss_x": 0.0, "loss_y": 0.0, "samples": 0}
    projection, projected = [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            raster = batch["raster"].to(device)
            condition = batch["condition"].to(device)
            target_x = batch["target_x"].to(device)
            target_y = batch["target_y"].to(device)
            collars = batch["collar"].to(device)
            rho_x, rho_y = pool_axes(model(raster, condition), config.alpha)
            loss, lx, ly = imitation_loss(rho_x, rho_y, target_x, target_y, collars)
            n = len(raster)
            totals["loss"] += loss.item() * n
            totals["loss_x"] += lx.item() * n
            totals["loss_y"] += ly.item() * n
            totals["samples"] += n
            if project and projected < config.projection_samples:
                keep = min(n, config.projection_samples - projected)
                meshes, _, _ = _project_batch(
                    dataset, batch["index"][:keep], rho_x[:keep], rho_y[:keep]
                )
                for mesh in meshes:
                    projection.append(
                        {
                            key: value
                            for key, value in mesh.metadata.items()
                            if key.endswith(("projection_l1", "projection_max"))
                        }
                    )
                projected += keep
    result = {key: value / totals["samples"] for key, value in totals.items() if key != "samples"}
    result["samples"] = totals["samples"]
    if projection:
        for key in projection[0]:
            result[key + "_mean"] = float(np.mean([row[key] for row in projection]))
            result[key + "_max"] = float(np.max([row[key] for row in projection]))
        result["projected_samples"] = len(projection)
    return result


def train_model(
    manifest_path,
    target_path,
    output,
    *,
    config=None,
    device=None,
    resume=None,
    allow_dirty=False,
    dataset_factory=TeacherDataset,
    initial_checkpoint=None,
    training_kind="teacher_imitation",
    target_version=TEACHER_VERSION,
):
    config = config or TrainingConfig()
    commit, dirty = _git_state()
    if dirty and not allow_dirty:
        raise ValueError(
            "Commit implementation changes before reproducible training, or allow dirty explicitly"
        )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = dataset_factory(manifest_path, target_path, "train")
    validation = dataset_factory(manifest_path, target_path, "validation")
    if train.dataset_id != validation.dataset_id:
        raise ValueError("Training and validation dataset IDs differ")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if initial_checkpoint is None:
        model = ResUNet(config.width).to(device)
    else:
        model, initial_metadata = load_model(initial_checkpoint, device=device)
        if (
            model.width != config.width
            or tuple(initial_metadata["raster_shape"]) != train.raster_shape
            or initial_metadata["pooling"]["alpha"] != config.alpha
        ):
            raise ValueError("Initial checkpoint architecture, raster or pooling differs")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, start_epoch, best = [], 0, math.inf
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        previous_config, current_config = dict(state["config"]), asdict(config)
        previous_epochs = previous_config.pop("epochs")
        current_epochs = current_config.pop("epochs")
        if (
            previous_config != current_config
            or state["dataset_id"] != train.dataset_id
            or state.get("target_id") != train.target_id
            or state.get("training_kind") != training_kind
            or current_epochs < state["epoch"]
            or previous_epochs < state["epoch"]
        ):
            raise ValueError("Resume state training configuration or dataset differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start_epoch, best = state["history"], state["epoch"], state["best"]
    started = perf_counter()
    for epoch in range(start_epoch, config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator)
        model.train()
        total, imitation_total, repair_total, seen = 0.0, 0.0, 0.0, 0
        for step, batch in enumerate(loader):
            raster = batch["raster"].to(device)
            condition = batch["condition"].to(device)
            target_x = batch["target_x"].to(device)
            target_y = batch["target_y"].to(device)
            collars = batch["collar"].to(device)
            rho_x, rho_y = pool_axes(model(raster, condition), config.alpha)
            imitation, _, _ = imitation_loss(rho_x, rho_y, target_x, target_y, collars)
            repair = torch.zeros((), device=device)
            if config.repair_weight and step % config.repair_every == 0:
                meshes, xc, yc = _project_batch(train, batch["index"], rho_x, rho_y)
                repair = repair_loss(rho_x, rho_y, meshes, x_collars=xc, y_collars=yc)
            loss = imitation + config.repair_weight * repair
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = len(raster)
            total += loss.item() * n
            imitation_total += imitation.item() * n
            repair_total += repair.item() * n
            seen += n
        validation_metrics = evaluate_imitation(model, validation, config, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": total / seen,
            "train_imitation": imitation_total / seen,
            "train_repair": repair_total / seen,
            "validation": validation_metrics,
        }
        history.append(row)
        training_metadata = {
            "version": TRAINING_VERSION,
            "epoch": epoch + 1,
            "config": asdict(config),
            "validation": validation_metrics,
            "training_kind": training_kind,
            "target_version": target_version,
            "target_id": train.target_id,
            "initial_checkpoint_sha256": (
                None if initial_checkpoint is None else _sha256(initial_checkpoint)
            ),
        }
        if validation_metrics["loss"] < best:
            best = validation_metrics["loss"]
            save_model(
                output / "best.pt",
                model,
                raster_shape=train.raster_shape,
                alpha=config.alpha,
                training_commit=commit,
                dataset_version=train.dataset_id,
                training_metadata=training_metadata,
            )
        torch.save(
            {
                "version": TRAINING_VERSION,
                "epoch": epoch + 1,
                "best": best,
                "config": asdict(config),
                "dataset_id": train.dataset_id,
                "target_id": train.target_id,
                "training_kind": training_kind,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            },
            output / "resume.pt",
        )
        report = {
            "training_version": TRAINING_VERSION,
            "training_kind": training_kind,
            "dataset_id": train.dataset_id,
            "teacher_targets_sha256": _sha256(target_path),
            "training_commit": commit,
            "dirty_at_start": dirty,
            "device": str(device),
            "config": asdict(config),
            "best_validation_loss": best,
            "history": history,
            "elapsed_seconds": perf_counter() - started,
            "provenance": provenance(),
        }
        _json(output / "training.json", report)
        print(
            f"epoch {epoch + 1}/{config.epochs}: train={row['train_imitation']:.6g} "
            f"validation={validation_metrics['loss']:.6g}",
            flush=True,
        )
    return report


def evaluate_checkpoint(
    manifest_path,
    target_path,
    checkpoint,
    output,
    *,
    split="test_iid",
    projection_samples=16,
    cuda_scenes=1,
    device=None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, metadata = load_model(checkpoint, device=device)
    dataset = TeacherDataset(manifest_path, target_path, split)
    config_values = metadata.get("training", {}).get("config", {})
    config = TrainingConfig(**config_values) if config_values else TrainingConfig()
    config = TrainingConfig(**{**asdict(config), "projection_samples": projection_samples})
    imitation = evaluate_imitation(model, dataset, config, device)
    physical = []
    seen = set()
    for record in dataset.records:
        scene = record["scene"]
        if scene.scene_id in seen or len(seen) >= cuda_scenes:
            continue
        seen.add(scene.scene_id)
        budget = record["budget"]
        evaluation = EvaluationConfig()
        times, frequencies = grids(scene, evaluation)
        teacher = run_scene(scene, budget, evaluation, strategy="heuristic")
        learned = run_scene(scene, budget, evaluation, strategy="cnn", checkpoint=checkpoint)
        a, b = sample_observables(teacher, times), sample_observables(learned, times)
        physical.append(
            {
                "scene_id": scene.scene_id,
                "budget": budget,
                "teacher_agreement": compare_observables(b, a, times, frequencies),
                "teacher_diagnostics": teacher.diagnostics,
                "cnn_diagnostics": learned.diagnostics,
            }
        )
    report = {
        "training_version": TRAINING_VERSION,
        "dataset_id": dataset.dataset_id,
        "split": split,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "imitation": imitation,
        "physical_teacher_agreement": physical,
        "note": "CUDA comparisons measure agreement with the heuristic teacher, not converged-reference accuracy.",
        "provenance": provenance(),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    targets = sub.add_parser("targets")
    targets.add_argument("--manifest", required=True)
    targets.add_argument("--output", required=True)
    targets.add_argument("--splits", nargs="+", default=["train", "validation", "test_iid"])
    train = sub.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--targets", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--width", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--repair-weight", type=float, default=0.05)
    train.add_argument("--repair-every", type=int, default=4)
    train.add_argument("--projection-samples", type=int, default=8)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--device")
    train.add_argument("--resume")
    train.add_argument("--allow-dirty", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--targets", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", default="test_iid")
    evaluate.add_argument("--projection-samples", type=int, default=16)
    evaluate.add_argument("--cuda-scenes", type=int, default=1)
    evaluate.add_argument("--device")
    args = parser.parse_args()
    if args.command == "targets":
        build_teacher_targets(args.manifest, args.output, splits=args.splits)
    elif args.command == "train":
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            width=args.width,
            learning_rate=args.learning_rate,
            repair_weight=args.repair_weight,
            repair_every=args.repair_every,
            projection_samples=args.projection_samples,
            seed=args.seed,
        )
        train_model(
            args.manifest,
            args.targets,
            args.output,
            config=config,
            device=args.device,
            resume=args.resume,
            allow_dirty=args.allow_dirty,
        )
    else:
        evaluate_checkpoint(
            args.manifest,
            args.targets,
            args.checkpoint,
            args.output,
            split=args.split,
            projection_samples=args.projection_samples,
            cuda_scenes=args.cuda_scenes,
            device=args.device,
        )


if __name__ == "__main__":
    main()
