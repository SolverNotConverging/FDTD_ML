"""Restartable set-valued profile training across curriculum stages."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .dataset_v2 import MeshProfileDatasetV2
from .model_v2 import AxisDensityUNetV2, set_valued_profile_loss


def curriculum_stages(epoch):
    """Short primitive warm-up, then cumulative training through separated scenes."""
    if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
        raise ValueError("Epoch must be a nonnegative integer")
    if epoch < 10:
        return frozenset(("C0",))
    return frozenset(("C0", "C1", "C2", "C3", "C5"))


def _atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _batch(dataset, indices, device):
    items = default_collate([dataset[int(index)] for index in indices])
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in items.items()
    }


def _loss(model, batch):
    prediction = model(batch["raster"], batch["conditioning"])
    return set_valued_profile_loss(
        prediction, batch["profiles"], batch["scores"], batch["best_index"], batch["candidate_mask"]
    )[0]


def probe_microbatch(base_channels, *, device="cuda:0", memory_limit_gib=20):
    """Select the largest tested FP16 microbatch below the stated memory cap."""
    if not torch.cuda.is_available():
        raise RuntimeError("The v2 training campaign requires a CUDA GPU")
    device = torch.device(device)
    torch.cuda.set_device(device)
    result = []
    for size in (1, 2, 4, 8):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = optimizer = raster = conditioning = prediction = loss = None
        try:
            model = AxisDensityUNetV2(base_channels=base_channels).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            raster = torch.rand(size, 9, 512, 512, device=device)
            conditioning = torch.rand(size, 15, device=device)
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(raster, conditioning)
                loss = prediction.cumsum(-1).square().mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            elapsed = time.perf_counter() - started
            result.append(
                {
                    "microbatch": size,
                    "peak_allocated_gib": peak,
                    "step_seconds": elapsed,
                    "admitted": peak < memory_limit_gib,
                }
            )
        except torch.cuda.OutOfMemoryError:
            result.append(
                {
                    "microbatch": size,
                    "peak_allocated_gib": None,
                    "step_seconds": None,
                    "admitted": False,
                }
            )
        finally:
            del model, optimizer, raster, conditioning, prediction, loss
            torch.cuda.empty_cache()
    admitted = [row["microbatch"] for row in result if row["admitted"]]
    if not admitted:
        raise RuntimeError("No FP16 microbatch fits the 20 GiB GPU memory limit")
    return max(admitted), result


def train_model(
    new_dataset,
    legacy_dataset,
    output,
    *,
    base_channels=64,
    device="cuda:0",
    deadline=None,
    microbatch=None,
    seed=20260923,
    max_epochs=60,
    curriculum_mode="legacy_cumulative",
    legacy_fraction=0.25,
):
    """Train one model with grouped splits and resumable C0 or cumulative schedules."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.cuda.set_device(device)
    new_dataset = Path(new_dataset)
    legacy_dataset = Path(legacy_dataset) if legacy_dataset is not None else None
    paths = [new_dataset] + ([legacy_dataset] if legacy_dataset is not None else [])
    inputs = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for dataset_path in paths
        for path in (dataset_path, dataset_path.parent / "targets.npz")
    }
    training = MeshProfileDatasetV2(paths, "train")
    validation = MeshProfileDatasetV2((new_dataset,), "validation")
    if curriculum_mode not in ("legacy_cumulative", "c0_restart"):
        raise ValueError("Unknown curriculum mode")
    if not 0 <= legacy_fraction < 1 or (legacy_dataset is None and legacy_fraction):
        raise ValueError("Legacy fraction needs a legacy dataset")
    if microbatch is None:
        microbatch, probe = probe_microbatch(base_channels, device=device)
    else:
        probe = [{"microbatch": microbatch, "admitted": "user_selected"}]
    if 32 % microbatch:
        raise ValueError("Microbatch must divide effective batch size 32")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = AxisDensityUNetV2(base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    state_path = output / "training_state.pt"
    epoch_start, best, stale, history = 0, float("inf"), 0, []
    if state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False)
        if (
            saved["inputs"] != inputs
            or saved["base_channels"] != base_channels
            or saved["microbatch"] != microbatch
            or saved.get("max_epochs", 60) != max_epochs
            or saved.get("curriculum_mode", "legacy_cumulative") != curriculum_mode
            or saved.get("legacy_fraction", 0.25) != legacy_fraction
        ):
            raise ValueError("Training resume fingerprint does not match")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        epoch_start, best, stale, history = (
            saved["next_epoch"],
            saved["best"],
            saved["stale"],
            saved["history"],
        )
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    _atomic_json(
        output / "launch.json",
        {
            "inputs": inputs,
            "base_channels": base_channels,
            "device": str(device),
            "microbatch": microbatch,
            "microbatch_probe": probe,
            "effective_batch": 32,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "precision": "FP16",
            "seed": seed,
            "max_epochs": max_epochs,
            "curriculum_mode": curriculum_mode,
            "legacy_fraction": legacy_fraction,
            "curriculum": {
                "foundation_warmup_epochs": 0 if curriculum_mode == "c0_restart" else 10,
                "cumulative_stages": ["C0"] if curriculum_mode == "c0_restart" else ["C0", "C1", "C2", "C3", "C5"],
                "cumulative_patience": 15 if curriculum_mode == "c0_restart" else 10,
            },
            "deadline_epoch_s": deadline,
        },
    )
    if curriculum_mode == "legacy_cumulative" and not 11 <= max_epochs <= 60:
        raise ValueError("The first cumulative curriculum runs for 11 to 60 epochs")
    if curriculum_mode == "c0_restart" and not 1 <= max_epochs <= 80:
        raise ValueError("The C0 restart runs for 1 to 80 epochs")
    for epoch in range(epoch_start, max_epochs):
        if deadline is not None and time.time() >= deadline:
            break
        stages = frozenset(("C0",)) if curriculum_mode == "c0_restart" else curriculum_stages(epoch)
        new = np.asarray(
            [
                i
                for i, ex in enumerate(training.examples)
                if ex["source"] == "new"
                and ex["split"] == "train"
                and ex["scene"]["stage"] in stages
            ]
        )
        old = np.asarray([i for i, ex in enumerate(training.examples) if ex["source"] == "legacy"])
        if len(new) == 0 or (legacy_fraction and len(old) == 0):
            raise ValueError("Curriculum needs new examples and requested qualified legacy examples")
        generator = np.random.default_rng(seed + epoch)
        draws = max(32, (len(new) + 31) // 32 * 32)
        old_draws = int(round(draws * legacy_fraction))
        indices = np.concatenate((generator.choice(new, draws - old_draws, replace=True),
                                  generator.choice(old, old_draws, replace=True) if old_draws else np.empty(0, dtype=int)))
        generator.shuffle(indices)
        model.train()
        running = []
        for offset in range(0, draws, 32):
            if deadline is not None and time.time() >= deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            for micro in range(offset, offset + 32, microbatch):
                batch = _batch(training, indices[micro : micro + microbatch], device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = _loss(model, batch)
                scaler.scale(loss / (32 // microbatch)).backward()
                running.append(float(loss.detach()))
            scaler.step(optimizer)
            scaler.update()
        if deadline is not None and time.time() >= deadline:
            break
        model.eval()
        losses = []
        with torch.no_grad():
            for offset in range(0, len(validation), microbatch):
                batch = _batch(
                    validation, range(offset, min(offset + microbatch, len(validation))), device
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    losses.append(float(_loss(model, batch)))
        value = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch + 1,
                "stage": "+".join(sorted(stages)),
                "training_profile_loss": float(np.mean(running)),
                "validation_profile_loss": value,
            }
        )
        if epoch == 10 and curriculum_mode == "legacy_cumulative":
            best, stale = float("inf"), 0
        if epoch >= 10 or curriculum_mode == "c0_restart":
            if value < best:
                best, stale = value, 0
            else:
                stale += 1
            checkpoint = output / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            _atomic_torch(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "base_channels": base_channels,
                    "epoch": epoch + 1,
                    "validation_profile_loss": value,
                    "inputs": inputs,
                },
            )
            checkpoints = sorted(
                output.glob("checkpoint_epoch_*.pt"),
                key=lambda p: torch.load(p, map_location="cpu", weights_only=False)[
                    "validation_profile_loss"
                ],
            )
            for obsolete in checkpoints[3:]:
                obsolete.unlink()
        _atomic_torch(
            state_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "next_epoch": epoch + 1,
                "best": best,
                "stale": stale,
                "history": history,
                "inputs": inputs,
                "base_channels": base_channels,
                "microbatch": microbatch,
                "max_epochs": max_epochs,
                "curriculum_mode": curriculum_mode,
                "legacy_fraction": legacy_fraction,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
        )
        _atomic_json(output / "history.json", history)
        if (epoch >= 10 or curriculum_mode == "c0_restart") and stale >= (15 if curriculum_mode == "c0_restart" else 10):
            break
    return {
        "epochs_completed": len(history),
        "best_validation_profile_loss": best,
        "deadline_reached": deadline is not None and time.time() >= deadline,
        "checkpoint_count": len(list(output.glob("checkpoint_epoch_*.pt"))),
    }
