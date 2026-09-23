#!/usr/bin/env python3
"""Train the first budget-conditioned axis-density model."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from scattermesh.distillation import probability_axis
from scattermesh.model import AxisDensityUNet, MeshDistillationDataset, set_valued_profile_loss


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_initial_weights(model, path, *, model_kwargs, input_schema):
    """Warm-start only a checkpoint with identical physical inputs and architecture."""
    initial = torch.load(path, map_location="cpu", weights_only=False)
    if initial.get("model") != "AxisDensityUNet" or initial.get("model_kwargs") != model_kwargs:
        raise ValueError("Initial checkpoint architecture does not match training config")
    if initial.get("config", {}).get("input_schema", "circle_v1") != input_schema:
        raise ValueError("Initial checkpoint input schema does not match training config")
    model.load_state_dict(initial["model_state"])


def source_hashes(dataset, config, initial_checkpoint=None):
    root = Path(__file__).resolve().parents[1]
    dataset = Path(dataset)
    metadata = json.loads(dataset.read_text())
    arrays = dataset.parent / metadata["arrays"]
    hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted((root / "src/scattermesh").glob("*.py"))
    }
    hashes[str(Path(__file__).resolve().relative_to(root))] = sha256_file(__file__)
    hashes["dataset"] = sha256_file(dataset)
    hashes["dataset_arrays"] = sha256_file(arrays)
    hashes["config"] = sha256_file(config)
    if initial_checkpoint is not None:
        hashes["initial_checkpoint"] = sha256_file(initial_checkpoint)
    return hashes


def acquire_training_lock(output):
    """Serialize trainers sharing one output directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / ".training.lock").open("a+")
    print(f"waiting for training lock: {output}", flush=True)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def completed_run_matches(output, provenance):
    """Return true only for a complete run produced from identical inputs."""
    output = Path(output)
    required = (
        output / "checkpoint.pt",
        output / "history.json",
        output / "predicted_meshes.json",
        output / "summary.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("status") == "complete"
        and summary.get("source_hashes") == provenance
        and summary.get("checkpoint_sha256") == sha256_file(output / "checkpoint.pt")
        and summary.get("predicted_meshes_sha256")
        == sha256_file(output / "predicted_meshes.json")
    )


def save_training_state(path, *, provenance, epoch, model, optimizer, generator,
                        history, best_validation, best_state, best_epoch, stale_epochs,
                        elapsed_seconds):
    """Save enough state to resume the next epoch after a worker interruption."""
    state = {
        "schema_version": 1,
        "source_hashes": provenance,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sampler_generator_state": generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "history": history,
        "best_validation": best_validation,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "elapsed_seconds": elapsed_seconds,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _family(example, family_weights):
    name = example["family"]
    if name.startswith("sparse"):
        return "sparse" if "sparse" in family_weights else name
    return "simple"


def _validation_weights(dataset, family_weights):
    if not family_weights:
        return None
    counts = {
        family: sum(_family(example, family_weights) == family for example in dataset.examples)
        for family in family_weights
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError("Every weighted validation family needs at least one example")
    weights = np.array(
        [family_weights[_family(example, family_weights)]
         / counts[_family(example, family_weights)] for example in dataset.examples],
        dtype=np.float64,
    )
    return torch.from_numpy((weights / weights.sum()).astype(np.float32))


def evaluate(model, loader, device, family_weights=None):
    model.eval()
    total, count = 0.0, 0
    weights = _validation_weights(loader.dataset, family_weights)
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["raster"].to(device), batch["conditioning"].to(device))
            loss, _ = set_valued_profile_loss(
                prediction,
                batch["profiles"].to(device),
                batch["scores"].to(device),
                batch["best_index"].to(device),
                batch["candidate_mask"].to(device),
                reduction="none" if weights is not None else "mean",
            )
            if weights is None:
                total += float(loss) * len(prediction)
            else:
                row_weights = weights[batch["sample_index"]].to(device)
                total += float((loss * row_weights).sum())
            count += len(prediction)
    return total if weights is not None else total / count


def evaluate_uniform(loader, device, family_weights=None):
    total, count = 0.0, 0
    weights = _validation_weights(loader.dataset, family_weights)
    with torch.no_grad():
        for batch in loader:
            profiles = batch["profiles"].to(device)
            prediction = torch.full(
                (len(profiles), 2, profiles.shape[-1]),
                1 / profiles.shape[-1],
                device=device,
            )
            loss, _ = set_valued_profile_loss(
                prediction,
                profiles,
                batch["scores"].to(device),
                batch["best_index"].to(device),
                batch["candidate_mask"].to(device),
                reduction="none" if weights is not None else "mean",
            )
            if weights is None:
                total += float(loss) * len(prediction)
            else:
                row_weights = weights[batch["sample_index"]].to(device)
                total += float((loss * row_weights).sum())
            count += len(prediction)
    return total if weights is not None else total / count


def projected_examples(model, dataset, device, max_ratio):
    model.eval()
    rows = []
    repairs = []
    with torch.no_grad():
        for index, example in enumerate(dataset.examples):
            batch = dataset[index]
            prediction = (
                model(batch["raster"][None].to(device), batch["conditioning"][None].to(device))[0]
                .cpu()
                .numpy()
            )
            x, x_repair = probability_axis(prediction[0], example["cells_x"], max_ratio=max_ratio)
            y, y_repair = probability_axis(prediction[1], example["cells_y"], max_ratio=max_ratio)
            repairs.extend((x_repair, y_repair))
            rows.append(
                {
                    "sample_id": example["sample_id"],
                    "split": example["split"],
                    "cells_x": example["cells_x"],
                    "cells_y": example["cells_y"],
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "x_uniform_repair_fraction": x_repair,
                    "y_uniform_repair_fraction": y_repair,
                }
            )
    return rows, {
        "mean_uniform_repair_fraction": float(np.mean(repairs)),
        "maximum_uniform_repair_fraction": float(np.max(repairs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/mesh_distillation_pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    provenance = source_hashes(args.dataset, args.config, args.init_checkpoint)
    _training_lock = acquire_training_lock(args.output)
    if completed_run_matches(args.output, provenance):
        print(f"matching completed training run already exists: {args.output}", flush=True)
        return
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    atomic_json(
        args.output / "launch.json",
        {
            "schema_version": 1,
            "status": "running",
            "dataset": str(args.dataset),
            "config_path": str(args.config),
            "config": config,
            "device": str(device),
            "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "source_hashes": provenance,
        },
    )

    raster_resolution = config.get("raster_resolution")
    input_schema = config.get("input_schema", "circle_v1")
    dataset_options = {"raster_resolution": raster_resolution, "input_schema": input_schema}
    train = MeshDistillationDataset(args.dataset, "train", **dataset_options)
    validation = MeshDistillationDataset(args.dataset, "validation", **dataset_options)
    test = MeshDistillationDataset(args.dataset, "test", **dataset_options)
    generator = torch.Generator().manual_seed(config["seed"])
    family_weights = config.get("family_sampling_weights")
    sampler = None
    if family_weights:
        allowed = ({"simple", "sparse"}, {"simple", "sparse_pair", "sparse_cluster"})
        if (set(family_weights) not in allowed
                or any(not np.isfinite(value) or value <= 0 for value in family_weights.values())
                or not np.isclose(sum(family_weights.values()), 1.0)):
            raise ValueError("Sampling weights require positive simple/sparse fractions summing to one")
        counts = {
            family: sum(_family(example, family_weights) == family for example in train.examples)
            for family in family_weights
        }
        if any(count == 0 for count in counts.values()):
            raise ValueError("Weighted training requires examples from every requested family")
        sample_weights = [
            family_weights[_family(example, family_weights)]
            / counts[_family(example, family_weights)]
            for example in train.examples
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train,
        batch_size=config.get("micro_batch_size", config["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        generator=None if sampler is not None else generator,
        num_workers=0,
    )
    micro_batch = config.get("micro_batch_size", config["batch_size"])
    effective_batch = config["batch_size"]
    if effective_batch % micro_batch:
        raise ValueError("Effective batch must be divisible by microbatch")
    validation_loader = DataLoader(validation, batch_size=micro_batch, num_workers=0)
    test_loader = DataLoader(test, batch_size=micro_batch, num_workers=0)
    uniform_validation_loss = evaluate_uniform(validation_loader, device, family_weights)
    uniform_test_loss = evaluate_uniform(test_loader, device, family_weights)
    model_kwargs = {
        "input_channels": 7 if input_schema == "sparse_v2" else 5,
        "conditioning_size": 15 if input_schema == "sparse_v2" else 10,
        "base_channels": config["base_channels"],
    }
    model = AxisDensityUNet(
        **model_kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history = []
    best_validation = np.inf
    best_state = None
    best_epoch = None
    stale_epochs = 0
    started = time.time()
    state_path = args.output / "training_state.pt"
    first_epoch = 1
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != 1 or state.get("source_hashes") != provenance:
            raise ValueError(f"Stale or incompatible training state: {state_path}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["sampler_generator_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if state["cuda_rng_states"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        history = state["history"]
        best_validation = state["best_validation"]
        best_state = state["best_state"]
        best_epoch = state["best_epoch"]
        stale_epochs = state["stale_epochs"]
        first_epoch = state["epoch"] + 1
        if stale_epochs >= config["patience"]:
            first_epoch = config["epochs"] + 1
        started -= state["elapsed_seconds"]
        print(f"resuming training at epoch {first_epoch}: {args.output}", flush=True)
    elif args.init_checkpoint is not None:
        load_initial_weights(
            model, args.init_checkpoint, model_kwargs=model_kwargs, input_schema=input_schema
        )
        print(f"initialized model from {args.init_checkpoint}", flush=True)
    for epoch in range(first_epoch, config["epochs"] + 1):
        model.train()
        train_values = []
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        for batch in train_loader:
            prediction = model(batch["raster"].to(device), batch["conditioning"].to(device))
            loss, _ = set_valued_profile_loss(
                prediction,
                batch["profiles"].to(device),
                batch["scores"].to(device),
                batch["best_index"].to(device),
                batch["candidate_mask"].to(device),
            )
            (loss * (len(prediction) / effective_batch)).backward()
            accumulated += len(prediction)
            if accumulated >= effective_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            train_values.append(float(loss.detach()))
        if accumulated:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss = float(np.mean(train_values))
        validation_loss = evaluate(model, validation_loader, device, family_weights)
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        history.append(row)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        save_training_state(
            state_path,
            provenance=provenance,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            generator=generator,
            history=history,
            best_validation=best_validation,
            best_state=best_state,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            elapsed_seconds=time.time() - started,
        )
        atomic_json(
            args.output / "progress.json",
            {
                "schema_version": 1,
                "status": "running",
                "epoch": epoch,
                "planned_epochs": config["epochs"],
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
                "stale_epochs": stale_epochs,
                "latest": row,
                "wall_seconds": time.time() - started,
            },
        )
        if epoch == 1 or epoch % config["report_every"] == 0:
            print(json.dumps(row), flush=True)
        if stale_epochs >= config["patience"]:
            break

    model.load_state_dict(best_state)
    test_loss = evaluate(model, test_loader, device, family_weights)
    meshes, repair_summary = projected_examples(
        model, validation, device, config["max_grading_ratio"]
    )
    test_meshes, test_repair = projected_examples(model, test, device, config["max_grading_ratio"])
    checkpoint = {
        "schema_version": 1,
        "model": "AxisDensityUNet",
        "model_kwargs": model_kwargs,
        "model_state": best_state,
        "dataset": str(args.dataset),
        "config": config,
        "source_hashes": provenance,
        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "test_loss": test_loss,
        "uniform_validation_loss": uniform_validation_loss,
        "uniform_test_loss": uniform_test_loss,
        "validation_improvement_over_uniform": uniform_validation_loss / best_validation,
        "test_improvement_over_uniform": uniform_test_loss / test_loss,
    }
    checkpoint_path = args.output / "checkpoint.pt"
    temporary = args.output / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    atomic_json(args.output / "history.json", {"epochs": history})
    predicted_meshes_path = args.output / "predicted_meshes.json"
    atomic_json(predicted_meshes_path, {"validation": meshes, "test": test_meshes})
    summary = {
        "schema_version": 1,
        "status": "complete",
        "parameter_count": parameter_count,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "initial_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "best_validation_loss": best_validation,
        "test_loss": test_loss,
        "source_hashes": provenance,
        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "predicted_meshes_sha256": sha256_file(predicted_meshes_path),
        "uniform_validation_loss": uniform_validation_loss,
        "uniform_test_loss": uniform_test_loss,
        "validation_improvement_over_uniform": uniform_validation_loss / best_validation,
        "test_improvement_over_uniform": uniform_test_loss / test_loss,
        "wall_seconds": time.time() - started,
        "validation_projection": repair_summary,
        "test_projection": test_repair,
    }
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "progress.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
