#!/usr/bin/env python3
"""Train the first budget-conditioned axis-density model."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scattermesh.distillation import probability_axis
from scattermesh.model import AxisDensityUNet, MeshDistillationDataset, set_valued_profile_loss


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(dataset, config):
    root = Path(__file__).resolve().parents[1]
    hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted((root / "src/scattermesh").glob("*.py"))
    }
    hashes[str(Path(__file__).resolve().relative_to(root))] = sha256_file(__file__)
    hashes["dataset"] = sha256_file(dataset)
    hashes["config"] = sha256_file(config)
    return hashes


def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["raster"].to(device), batch["conditioning"].to(device))
            loss, _ = set_valued_profile_loss(
                prediction,
                batch["profiles"].to(device),
                batch["scores"].to(device),
                batch["best_index"].to(device),
                batch["candidate_mask"].to(device),
            )
            total += float(loss) * len(prediction)
            count += len(prediction)
    return total / count


def evaluate_uniform(loader, device):
    total, count = 0.0, 0
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
            )
            total += float(loss) * len(prediction)
            count += len(prediction)
    return total / count


def projected_examples(model, dataset, device, max_ratio):
    model.eval()
    rows = []
    repairs = []
    with torch.no_grad():
        for index, example in enumerate(dataset.examples):
            batch = dataset[index]
            prediction = model(
                batch["raster"][None].to(device), batch["conditioning"][None].to(device)
            )[0].cpu().numpy()
            x, x_repair = probability_axis(
                prediction[0], example["cells_x"], max_ratio=max_ratio
            )
            y, y_repair = probability_axis(
                prediction[1], example["cells_y"], max_ratio=max_ratio
            )
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
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    provenance = source_hashes(args.dataset, args.config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "launch.json",
        {
            "schema_version": 1,
            "status": "running",
            "dataset": str(args.dataset),
            "config_path": str(args.config),
            "config": config,
            "device": str(device),
            "source_hashes": provenance,
        },
    )

    train = MeshDistillationDataset(args.dataset, "train")
    validation = MeshDistillationDataset(args.dataset, "validation")
    test = MeshDistillationDataset(args.dataset, "test")
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(validation, batch_size=config["batch_size"], num_workers=0)
    test_loader = DataLoader(test, batch_size=config["batch_size"], num_workers=0)
    uniform_validation_loss = evaluate_uniform(validation_loader, device)
    uniform_test_loss = evaluate_uniform(test_loader, device)
    model = AxisDensityUNet(base_channels=config["base_channels"]).to(device)
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
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_values = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["raster"].to(device), batch["conditioning"].to(device))
            loss, _ = set_valued_profile_loss(
                prediction,
                batch["profiles"].to(device),
                batch["scores"].to(device),
                batch["best_index"].to(device),
                batch["candidate_mask"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            train_values.append(float(loss.detach()))
        train_loss = float(np.mean(train_values))
        validation_loss = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        history.append(row)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
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
    test_loss = evaluate(model, test_loader, device)
    meshes, repair_summary = projected_examples(
        model, validation, device, config["max_grading_ratio"]
    )
    test_meshes, test_repair = projected_examples(model, test, device, config["max_grading_ratio"])
    checkpoint = {
        "schema_version": 1,
        "model": "AxisDensityUNet",
        "model_kwargs": {"base_channels": config["base_channels"]},
        "model_state": best_state,
        "dataset": str(args.dataset),
        "config": config,
        "source_hashes": provenance,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "test_loss": test_loss,
        "uniform_validation_loss": uniform_validation_loss,
        "uniform_test_loss": uniform_test_loss,
        "validation_improvement_over_uniform": uniform_validation_loss / best_validation,
        "test_improvement_over_uniform": uniform_test_loss / test_loss,
    }
    temporary = args.output / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output / "checkpoint.pt")
    atomic_json(args.output / "history.json", {"epochs": history})
    atomic_json(
        args.output / "predicted_meshes.json",
        {"validation": meshes, "test": test_meshes},
    )
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
