"""Portable curriculum mesh targets and verified legacy-data imports."""

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .candidates_v2 import CANDIDATE_NAMES
from .curriculum_v2 import conditioning_v2, rasterize_v2
from .profiles_v2 import axis_probability, resample_axis_profiles
from .scoring_v2 import rank_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_dataset(destination, examples, profiles, scores, mask, provenance):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = destination / "targets.npz"
    temporary = destination / "targets.tmp.npz"
    np.savez_compressed(temporary, profiles=profiles, scores=scores, candidate_mask=mask)
    os.replace(temporary, arrays)
    metadata = {
        "schema_version": 3,
        "profile_bins": 512,
        "candidate_count": int(scores.shape[1]),
        "examples": examples,
        "arrays": arrays.name,
        "arrays_sha256": _sha256(arrays),
        "provenance": provenance,
    }
    temporary = destination / "dataset.tmp.json"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, destination / "dataset.json")
    return destination / "dataset.json"


def build_legacy_import(source_dataset, campaign_output, destination):
    """Import only old training examples after recomputing raw/dt teacher scores."""
    source_dataset = Path(source_dataset)
    campaign_output = Path(campaign_output)
    metadata = json.loads(source_dataset.read_text())
    if metadata.get("ranking", {}).get("mode") != "fixed_axis_soft_nt":
        raise ValueError("Unsupported legacy teacher ranking")
    with np.load(source_dataset.parent / metadata["arrays"]) as arrays:
        old_profiles = arrays["profiles"].copy()
        old_masks = arrays["candidate_mask"].copy()
    candidates = metadata["candidate_names"]
    examples, profiles, scores, masks = [], [], [], []
    for index, old in enumerate(metadata["examples"]):
        if old["split"] != "train" or old.get("shape") != "circle" or "objects" in old:
            continue
        records = []
        for name in candidates:
            path = campaign_output / "cases" / f"{old['sample_id']}_{name}" / "record.json"
            if not path.exists():
                raise ValueError(f"Legacy source record is missing: {path}")
            record = json.loads(path.read_text())
            records.append(
                {
                    "name": name,
                    "accepted": bool(record.get("accepted")),
                    "joint_scattering_loss": record.get("joint_scattering_loss"),
                    "dt": record.get("dt"),
                }
            )
        ranked = rank_candidates(records)
        best_name = ranked[0]["name"]
        new_scores = np.ones(len(candidates), dtype=np.float64)
        new_mask = np.array([item["accepted"] for item in records], dtype=bool)
        if not np.array_equal(new_mask, old_masks[index]):
            raise ValueError(f"Legacy validity mask changed: {old['sample_id']}")
        for item in ranked:
            new_scores[candidates.index(item["name"])] = item["teacher_score"]
        scene = {
            "schema_version": 2,
            "lineage_id": old["lineage_id"],
            "shape": "circle",
            "family": "legacy_circle",
            "stage": "C0",
            "center_m": old["center_m"],
            "radius_m": old["radius_m"],
            "feature_size_m": old["feature_size_m"],
            "material": {
                "kind": "dielectric",
                "epsilon_r": old["epsilon_r"],
                "sigma_e_s_per_m": old["sigma_e_s_per_m"],
            },
            "split": "train",
        }
        examples.append(
            {
                "sample_id": old["sample_id"],
                "scene": scene,
                "source": "legacy",
                "split": "train",
                "cells_x": old["cells_x"],
                "cells_y": old["cells_y"],
                "incidence_angle_rad": old["incidence_angle_rad"],
                "frequencies_hz": old["frequencies_hz"],
                "best_candidate_index": candidates.index(best_name),
            }
        )
        profiles.append(old_profiles[index])
        scores.append(new_scores)
        masks.append(new_mask)
    if not examples:
        raise ValueError("No accepted legacy training examples found")
    return _save_dataset(
        destination,
        examples,
        resample_axis_profiles(np.asarray(profiles, dtype=np.float32), 512),
        np.asarray(scores),
        np.asarray(masks),
        {
            "kind": "legacy_re_ranked",
            "source_dataset_sha256": _sha256(source_dataset),
            "source_dataset": _portable_path(source_dataset),
            "source_arrays_sha256": _sha256(source_dataset.parent / metadata["arrays"]),
            "source_campaign": _portable_path(campaign_output),
            "time_step_exponent": 0.05,
        },
    )


def build_new_dataset(manifest_path, campaign_output, destination):
    """Create train/validation targets; frozen test scenes never become labels."""
    manifest_path = Path(manifest_path)
    campaign_output = Path(campaign_output)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") in ("compact_c8_c9_poc_v1", "c0_restart_v1"):
        from .campaign_v2 import scoring_fingerprint

        if manifest.get("scoring_fingerprint") != scoring_fingerprint():
            raise ValueError("Scoring implementation changed after compact campaign freeze")
    candidate_names = tuple(manifest.get("candidate_names", CANDIDATE_NAMES))
    examples, profiles, scores, masks = [], [], [], []
    excluded = []
    for scene in manifest["scenes"]:
        if scene["split"] == "test":
            continue
        conditions = manifest.get("conditions_by_scene", {}).get(scene["lineage_id"])
        if conditions is None:
            conditions = [
                {"angle_index": angle_index, "cells": cells}
                for angle_index, angle in enumerate(manifest["incidence_angles"])
                for cells in manifest["budgets"]
            ]
        for condition in conditions:
            angle_index, cells = condition["angle_index"], condition["cells"]
            angle = manifest["incidence_angles"][angle_index]
            qualification = (
                campaign_output / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.json"
            )
            if not qualification.exists() or not json.loads(qualification.read_text()).get(
                "accepted"
            ):
                excluded.append(
                    {
                        "scene_id": scene["lineage_id"],
                        "angle_index": angle_index,
                        "reason": "reference_not_qualified",
                    }
                )
                continue
            rows, axes = [], []
            for name in candidate_names:
                directory = (
                    campaign_output
                    / "cases"
                    / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
                )
                record_path, arrays_path = directory / "record.json", directory / "spectra.npz"
                if not record_path.exists():
                    rows = []
                    break
                record = json.loads(record_path.read_text())
                rows.append(
                    {
                        "name": name,
                        "accepted": bool(record.get("accepted")),
                        "status": record.get("status"),
                        "joint_scattering_loss": record.get("joint_scattering_loss"),
                        "dt": record.get("dt"),
                    }
                )
                if arrays_path.exists():
                    with np.load(arrays_path) as arrays:
                        axes.append((arrays["x"].copy(), arrays["y"].copy()))
                else:
                    axes.append(None)
            if len(rows) != len(candidate_names):
                excluded.append(
                    {
                        "scene_id": scene["lineage_id"],
                        "angle_index": angle_index,
                        "cells": cells,
                        "reason": "candidate_record_incomplete",
                    }
                )
                continue
            if any(row["status"] == "incomplete" for row in rows):
                excluded.append(
                    {
                        "scene_id": scene["lineage_id"],
                        "angle_index": angle_index,
                        "cells": cells,
                        "reason": "candidate_execution_incomplete",
                    }
                )
                continue
            try:
                ranked = rank_candidates(rows)
            except ValueError:
                excluded.append(
                    {
                        "scene_id": scene["lineage_id"],
                        "angle_index": angle_index,
                        "cells": cells,
                        "reason": "uniform_invalid",
                    }
                )
                continue
            row_scores = np.ones(len(rows), dtype=np.float64)
            row_mask = np.array([row["accepted"] for row in rows])
            row_profiles = np.full((len(rows), 2, 512), 1 / 512, dtype=np.float32)
            for candidate in ranked:
                index = candidate_names.index(candidate["name"])
                if (
                    axes[index] is None
                    or len(axes[index][0]) - 1 != cells
                    or len(axes[index][1]) - 1 != cells
                ):
                    raise ValueError("Accepted candidate lacks exact-budget saved axes")
                row_scores[index] = candidate["teacher_score"]
                row_profiles[index, 0] = axis_probability(axes[index][0], 512)
                row_profiles[index, 1] = axis_probability(axes[index][1], 512)
            examples.append(
                {
                    "sample_id": f"{scene['lineage_id']}_a{angle_index}_n{cells}",
                    "scene": scene,
                    "source": "new",
                    "split": scene["split"],
                    "cells_x": cells,
                    "cells_y": cells,
                    "incidence_angle_rad": angle,
                    "frequencies_hz": manifest["frequencies_hz"],
                    "best_candidate_index": candidate_names.index(ranked[0]["name"]),
                }
            )
            profiles.append(row_profiles)
            scores.append(row_scores)
            masks.append(row_mask)
    if not examples:
        raise ValueError("No reference-qualified training examples found")
    path = _save_dataset(
        destination,
        examples,
        np.asarray(profiles),
        np.asarray(scores),
        np.asarray(masks),
        {
            "kind": "new_curriculum_v2",
            "manifest_sha256": _sha256(manifest_path),
            "excluded": excluded,
            "time_step_exponent": 0.05,
        },
    )
    return path


class MeshProfileDatasetV2(Dataset):
    """Lazy 512-pixel rasterization with bounded per-process geometry cache."""

    def __init__(self, paths, split, *, cache_size=32):
        self.examples, target_profiles, target_scores, target_masks = [], [], [], []
        for path in map(Path, paths):
            metadata = json.loads(path.read_text())
            arrays_path = path.parent / metadata["arrays"]
            if _sha256(arrays_path) != metadata["arrays_sha256"]:
                raise ValueError(f"Dataset array checksum mismatch: {arrays_path}")
            with np.load(arrays_path) as arrays:
                indices = [i for i, ex in enumerate(metadata["examples"]) if ex["split"] == split]
                self.examples.extend(metadata["examples"][i] for i in indices)
                target_profiles.extend(arrays["profiles"][indices])
                target_scores.extend(arrays["scores"][indices])
                target_masks.extend(arrays["candidate_mask"][indices])
        if not self.examples:
            raise ValueError(f"No examples for split {split}")
        candidates = max(len(row) for row in target_scores)
        self.profiles = np.full((len(self.examples), candidates, 2, 512), 1 / 512, dtype=np.float32)
        self.scores = np.ones((len(self.examples), candidates), dtype=np.float32)
        self.mask = np.zeros((len(self.examples), candidates), dtype=bool)
        for index, (p, s, m) in enumerate(zip(target_profiles, target_scores, target_masks)):
            self.profiles[index, : len(p)] = p
            self.scores[index, : len(s)] = s
            self.mask[index, : len(m)] = m
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        scene = example["scene"]
        key = scene["lineage_id"]
        if key not in self.cache:
            self.cache[key] = rasterize_v2(scene)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return {
            "raster": torch.from_numpy(self.cache[key]),
            "conditioning": torch.from_numpy(
                conditioning_v2(
                    scene,
                    example["cells_x"],
                    example["cells_y"],
                    example["incidence_angle_rad"],
                    example["frequencies_hz"],
                )
            ),
            "profiles": torch.from_numpy(self.profiles[index]),
            "scores": torch.from_numpy(self.scores[index]),
            "candidate_mask": torch.from_numpy(self.mask[index]),
            "best_index": torch.tensor(example["best_candidate_index"], dtype=torch.long),
            "source": example["source"],
            "stage": scene["stage"],
        }
