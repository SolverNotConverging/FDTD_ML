"""Frozen C0--C2 manifest and restartable independent-GPU data workers."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .candidates_v2 import CANDIDATE_NAMES
from .curriculum_v2 import generate_lineages
from .qualification_v2 import qualify_reference
from .scoring_v2 import rank_candidates
from .simulation_v2 import FREQUENCIES, evaluate_case, numerical_source_hashes

INCIDENCE_ANGLES = (0.0, float(np.pi / 3))
BUDGETS = (32, 48, 64, 96)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def freeze_manifest(profile_path, output, *, seed=20260923):
    """Fix corpus size and geometry lineages after the measured resource gate."""
    profile = json.loads(Path(profile_path).read_text())
    count = profile.get("selected_lineages")
    if profile.get("gate") != "ready_for_bulk" or count not in (128, 256):
        raise ValueError("Measured profile does not admit a 128/256-lineage campaign")
    output = Path(output)
    profile_hash = hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "lineage_count": count,
        "seed": seed,
        "source_profile_sha256": profile_hash,
        "frequencies_hz": FREQUENCIES,
        "incidence_angles": INCIDENCE_ANGLES,
        "budgets": BUDGETS,
        "scenes": generate_lineages(count, seed=seed),
        "numerical_source_hashes": numerical_source_hashes(),
    }
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != manifest:
            raise ValueError("Frozen campaign manifest differs from requested values")
    else:
        atomic_json(output, manifest)
    return manifest


def _read_axes(directory):
    with np.load(directory / "spectra.npz") as arrays:
        return arrays["x"].copy(), arrays["y"].copy()


def _incomplete_case(directory, scene, angle, cells, policy):
    path = directory / "record.json"
    if path.exists():
        return
    atomic_json(
        path,
        {
            "schema_version": 2,
            "scene_id": scene["lineage_id"],
            "incidence_angle_rad": angle,
            "cells_x": cells,
            "cells_y": cells,
            "policy": policy,
            "status": "incomplete",
            "accepted": False,
            "reason": "campaign_deadline",
        },
    )


def run_data_worker(manifest_path, output, *, shard, shards=4, device="cuda:0", deadline=None):
    """One scene stays on one worker, avoiding duplicate CUDA/reference writes."""
    manifest = json.loads(Path(manifest_path).read_text())
    if not 0 <= shard < shards:
        raise ValueError("Shard must be within the worker count")
    output = Path(output)
    source_hashes = numerical_source_hashes()
    if manifest["numerical_source_hashes"] != source_hashes:
        raise ValueError("Numerical implementation changed after dataset freeze")
    summary = {
        "shard": shard,
        "device": device,
        "references_accepted": 0,
        "references_rejected": 0,
        "cases_accepted": 0,
        "cases_other": 0,
        "cases_incomplete": 0,
        "started_at": time.time(),
    }
    for scene_index, scene in enumerate(manifest["scenes"]):
        if scene_index % shards != shard:
            continue
        for angle_index, angle in enumerate(manifest["incidence_angles"]):
            if deadline is not None and time.time() >= deadline:
                summary["deadline_reached"] = True
                atomic_json(output / "workers" / f"shard_{shard}.json", summary)
                return summary
            qualified = qualify_reference(
                scene, angle_index, angle, output, device=device, deadline=deadline
            )
            if not qualified.get("accepted"):
                summary["references_rejected"] += 1
                continue
            summary["references_accepted"] += 1
            reference_path = output / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.npz"
            with np.load(reference_path) as arrays:
                reference = arrays["complex_far_field"].copy()
            for cells in manifest["budgets"]:
                rows, directories = [], {}
                for name in CANDIDATE_NAMES[:9]:
                    directory = (
                        output / "cases" / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
                    )
                    directories[name] = directory
                    if deadline is not None and time.time() >= deadline:
                        _incomplete_case(directory, scene, angle, cells, name)
                        summary["cases_incomplete"] += 1
                        continue
                    record, _ = evaluate_case(
                        scene,
                        cells,
                        angle,
                        name,
                        directory,
                        device=device,
                        reference=reference,
                        source_hashes=source_hashes,
                    )
                    rows.append({"name": name, **record})
                    summary["cases_accepted" if record.get("accepted") else "cases_other"] += 1
                try:
                    best = rank_candidates(rows)[0]
                    best_axes = _read_axes(directories[best["name"]])
                except (ValueError, IndexError, KeyError, OSError):
                    best_axes = None
                for name in CANDIDATE_NAMES[9:]:
                    directory = (
                        output / "cases" / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
                    )
                    if deadline is not None and time.time() >= deadline:
                        _incomplete_case(directory, scene, angle, cells, name)
                        summary["cases_incomplete"] += 1
                        continue
                    if best_axes is None:
                        atomic_json(
                            directory / "record.json",
                            {
                                "schema_version": 2,
                                "scene_id": scene["lineage_id"],
                                "cells_x": cells,
                                "cells_y": cells,
                                "policy": name,
                                "status": "no_valid_seed",
                                "accepted": False,
                            },
                        )
                        summary["cases_other"] += 1
                        continue
                    record, _ = evaluate_case(
                        scene,
                        cells,
                        angle,
                        name,
                        directory,
                        device=device,
                        reference=reference,
                        selected_axes=best_axes,
                        source_hashes=source_hashes,
                    )
                    summary["cases_accepted" if record.get("accepted") else "cases_other"] += 1
            atomic_json(output / "workers" / f"shard_{shard}.json", summary)
    summary["finished_at"] = time.time()
    atomic_json(output / "workers" / f"shard_{shard}.json", summary)
    return summary


def mark_unfinished(manifest_path, output):
    """Materialize deadline-interrupted work as resumable incomplete records."""
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(output)
    counts = {"references": 0, "cases": 0}
    for scene in manifest["scenes"]:
        for angle_index, angle in enumerate(manifest["incidence_angles"]):
            qualification = output / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.json"
            if not qualification.exists():
                atomic_json(
                    qualification,
                    {
                        "schema_version": 2,
                        "terminal": False,
                        "accepted": False,
                        "status": "incomplete",
                        "reason": "campaign_deadline",
                        "scene_id": scene["lineage_id"],
                        "angle_index": angle_index,
                    },
                )
                counts["references"] += 1
            for cells in manifest["budgets"]:
                for name in CANDIDATE_NAMES:
                    directory = (
                        output / "cases" / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
                    )
                    if not (directory / "record.json").exists():
                        _incomplete_case(directory, scene, angle, cells, name)
                        counts["cases"] += 1
    atomic_json(output / "incomplete_counts.json", counts)
    return counts
