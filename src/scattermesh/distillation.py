"""Training-data preparation and exact-axis projection for learned mesh densities."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

DOMAIN = 1.2


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def axis_probability(axis, bins, *, length=DOMAIN):
    """Convert grid nodes to probability mass on fixed uniform spatial bins.

    Every mesh cell contributes equal mass. Within a cell that mass is distributed
    uniformly, so inverse-CDF projection approximately reconstructs the source axis.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if (
        axis.ndim != 1
        or len(axis) < 5
        or not np.isfinite(axis).all()
        or axis[0] != 0
        or not np.isclose(axis[-1], length)
        or np.any(np.diff(axis) <= 0)
        or isinstance(bins, bool)
        or int(bins) != bins
        or bins < 8
    ):
        raise ValueError("A finite increasing axis and at least eight profile bins are required")
    edges = np.linspace(0.0, length, int(bins) + 1)
    cells = len(axis) - 1
    indices = np.searchsorted(axis, edges, side="right") - 1
    indices = np.clip(indices, 0, cells - 1)
    fractions = (edges - axis[indices]) / np.diff(axis)[indices]
    cumulative = (indices + np.clip(fractions, 0.0, 1.0)) / cells
    cumulative[0], cumulative[-1] = 0.0, 1.0
    probability = np.diff(cumulative)
    probability = np.maximum(probability, 0.0)
    return probability / probability.sum()


def _quantile_axis(probability, cells, length):
    probability = np.asarray(probability, dtype=np.float64)
    probability = np.maximum(probability, np.finfo(np.float64).tiny)
    probability /= probability.sum()
    cumulative = np.r_[0.0, np.cumsum(probability)]
    cumulative[-1] = 1.0
    quantiles = np.linspace(0.0, 1.0, int(cells) + 1)
    indices = np.searchsorted(cumulative, quantiles, side="right") - 1
    indices = np.clip(indices, 0, len(probability) - 1)
    fractions = (quantiles - cumulative[indices]) / probability[indices]
    axis = (indices + np.clip(fractions, 0.0, 1.0)) * length / len(probability)
    axis[0], axis[-1] = 0.0, float(length)
    return axis


def _axis_grading(axis):
    spacing = np.diff(axis)
    return float(max(np.max(spacing[1:] / spacing[:-1]), np.max(spacing[:-1] / spacing[1:])))


def probability_axis(probability, cells, *, length=DOMAIN, max_ratio=3.0):
    """Project a positive profile to exactly ``cells`` intervals with a grading cap.

    If the raw inverse-CDF axis exceeds the cap, the profile is minimally mixed
    toward uniform by bisection. The returned repair fraction is auditable and is
    zero when no repair was needed.
    """
    probability = np.asarray(probability, dtype=np.float64)
    if (
        probability.ndim != 1
        or len(probability) < 8
        or not np.isfinite(probability).all()
        or np.any(probability < 0)
        or probability.sum() <= 0
        or isinstance(cells, bool)
        or int(cells) != cells
        or cells < 4
        or not np.isfinite(length)
        or length <= 0
        or not np.isfinite(max_ratio)
        or max_ratio < 1
    ):
        raise ValueError("Invalid probability profile, cell count, domain, or grading cap")
    probability = probability / probability.sum()
    raw = _quantile_axis(probability, cells, length)
    if _axis_grading(raw) <= max_ratio * (1 + 1e-12):
        return raw, 0.0
    uniform = np.full_like(probability, 1 / len(probability))
    low, high = 0.0, 1.0
    accepted = np.linspace(0.0, length, int(cells) + 1)
    for _ in range(48):
        middle = (low + high) / 2
        candidate = _quantile_axis((1 - middle) * probability + middle * uniform, cells, length)
        if _axis_grading(candidate) <= max_ratio * (1 + 1e-12):
            high, accepted = middle, candidate
        else:
            low = middle
    return accepted, high


def rasterize_circle(geometry, resolution=128, *, domain=DOMAIN):
    """Rasterize a continuous circle into material, SDF, and interface channels."""
    if geometry.get("shape") != "circle":
        raise ValueError(f"Unsupported learning geometry: {geometry.get('shape')}")
    if isinstance(resolution, bool) or int(resolution) != resolution or resolution < 16:
        raise ValueError("resolution must be an integer >=16")
    center = np.asarray(geometry["center_m"], dtype=np.float64)
    radius = float(geometry["radius_m"])
    if center.shape != (2,) or not np.isfinite(center).all() or radius <= 0:
        raise ValueError("Circle needs a finite center and positive radius")
    coordinate = (np.arange(int(resolution)) + 0.5) * domain / resolution
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    signed_distance = radius - np.hypot(x - center[0], y - center[1])
    occupancy = (signed_distance >= 0).astype(np.float32)
    epsilon = float(geometry["epsilon_r"])
    sigma = float(geometry["sigma_e_s_per_m"])
    pixel = domain / resolution
    channels = np.stack(
        [
            occupancy,
            occupancy * np.log(epsilon) / np.log(30.0),
            occupancy * np.log1p(sigma / 0.01) / np.log1p(0.35 / 0.01),
            np.clip(signed_distance / max(0.25 * geometry["feature_size_m"], pixel), -1, 1),
            np.exp(-np.abs(signed_distance) / (2 * pixel)),
        ]
    )
    return channels.astype(np.float32)


def conditioning_features(example, *, domain=DOMAIN, max_cells=128, max_ratio=3.0):
    """Return global source, budget, band, feature, and material conditioning."""
    frequencies = np.asarray(example["frequencies_hz"], dtype=np.float64)
    angle = float(example["incidence_angle_rad"])
    values = np.array(
        [
            np.sin(angle),
            np.cos(angle),
            example["cells_x"] / max_cells,
            example["cells_y"] / max_cells,
            frequencies.min() / 1.2e9,
            frequencies.max() / 1.2e9,
            example["feature_size_m"] / domain,
            np.log(example["epsilon_r"]) / np.log(30.0),
            np.log1p(example["sigma_e_s_per_m"] / 0.01) / np.log1p(0.35 / 0.01),
            max_ratio / 3.0,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("Conditioning features must be finite")
    return values


def build_distillation_dataset(
    manifest_path,
    campaign_path,
    campaign_output,
    destination,
    *,
    profile_bins=128,
):
    """Validate a completed campaign and materialize set-valued axis targets."""
    manifest_path, campaign_path = Path(manifest_path), Path(campaign_path)
    campaign_output, destination = Path(campaign_output), Path(destination)
    manifest = json.loads(manifest_path.read_text())
    campaign = json.loads(campaign_path.read_text())
    report = json.loads((campaign_output / "report.json").read_text())
    labels = json.loads((campaign_output / "labels.json").read_text())["groups"]
    if campaign["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Campaign and manifest dataset IDs differ")
    if report.get("campaign_id") != campaign["campaign_id"] or report.get("decision") != "accepted":
        raise ValueError("Training data require the accepted report for this exact campaign")
    readiness = report.get("training_readiness", {})
    if readiness.get("decision") != "ready_for_m5_pilot" or not all(
        readiness.get("checks", {}).values()
    ):
        raise ValueError("Training data require a passing pre-M5 label-diversity gate")
    expected_cases = len(campaign["condition_ids"]) * len(campaign["candidate_names"])
    status_counts = report.get("status_counts", {})
    if report.get("case_count") != expected_cases or sum(status_counts.values()) != expected_cases:
        raise ValueError("Training data require one terminal record per campaign case")
    ranking = campaign["ranking"]
    if ranking.get("mode") != "fixed_axis_soft_nt":
        raise ValueError("Distillation requires exact-axis soft-Nt labels")

    geometries = {row["geometry_id"]: row for row in manifest["geometries"]}
    conditions = {row["task_id"]: row for row in manifest["conditions"]}
    candidates = tuple(campaign["candidate_names"])
    profiles = np.zeros(
        (len(campaign["condition_ids"]), len(candidates), 2, int(profile_bins)),
        dtype=np.float32,
    )
    scores = np.zeros((len(campaign["condition_ids"]), len(candidates)), dtype=np.float64)
    candidate_mask = np.zeros((len(campaign["condition_ids"]), len(candidates)), dtype=bool)
    examples = []
    split_counts = Counter()
    exponent = float(ranking["nt_cost_exponent"])
    for sample_index, condition_id in enumerate(campaign["condition_ids"]):
        condition = conditions[condition_id]
        geometry = geometries[condition["geometry_id"]]
        records = {}
        axes = {}
        for candidate_index, candidate in enumerate(candidates):
            case_id = f"{condition_id}_{candidate}"
            directory = campaign_output / "cases" / case_id
            record = json.loads((directory / "record.json").read_text())
            if record["config"]["cells"] != condition["cells_x"]:
                raise ValueError(f"Invalid exact-budget training case: {case_id}")
            with np.load(directory / "spectra.npz") as arrays:
                x, y = arrays["x"].copy(), arrays["y"].copy()
            if len(x) - 1 != condition["cells_x"] or len(y) - 1 != condition["cells_y"]:
                raise ValueError(f"Axis count differs from requested budget: {case_id}")
            records[candidate], axes[candidate] = record, (x, y)
            candidate_mask[sample_index, candidate_index] = bool(record.get("accepted"))
        baseline = records["uniform"]
        if not baseline.get("accepted"):
            raise ValueError(f"Training condition lacks an accepted uniform baseline: {condition_id}")
        for candidate_index, candidate in enumerate(candidates):
            record = records[candidate]
            if record.get("accepted"):
                scores[sample_index, candidate_index] = record["joint_scattering_loss"] * (
                    record["Nt"] / baseline["Nt"]
                ) ** exponent
            else:
                scores[sample_index, candidate_index] = 1.0
            profiles[sample_index, candidate_index, 0] = axis_probability(
                axes[candidate][0], profile_bins
            )
            profiles[sample_index, candidate_index, 1] = axis_probability(
                axes[candidate][1], profile_bins
            )
        ranked_scores = np.where(candidate_mask[sample_index], scores[sample_index], np.inf)
        best_index = int(np.argmin(ranked_scores))
        budget_label = labels[condition["illumination_id"]]["budgets"][
            str(condition["cells_x"])
        ]
        expected_best = f"{condition_id}_{candidates[best_index]}"
        if budget_label["best_case"] != expected_best:
            raise ValueError(f"Saved label ranking differs for {condition_id}")
        example = {
            "sample_id": condition_id,
            "geometry_id": condition["geometry_id"],
            "lineage_id": condition["lineage_id"],
            "illumination_id": condition["illumination_id"],
            "split": condition["split"],
            "family": condition["family"],
            "shape": geometry["shape"],
            "radius_m": geometry["radius_m"],
            "center_m": geometry["center_m"],
            "feature_size_m": geometry["feature_size_m"],
            "epsilon_r": geometry["epsilon_r"],
            "sigma_e_s_per_m": geometry["sigma_e_s_per_m"],
            "incidence_angle_rad": condition["incidence_angle_rad"],
            "frequencies_hz": geometry["frequencies_hz"],
            "cells_x": condition["cells_x"],
            "cells_y": condition["cells_y"],
            "best_candidate": candidates[best_index],
            "best_candidate_index": best_index,
            "uniform_score": float(scores[sample_index, candidates.index("uniform")]),
            "best_score": float(scores[sample_index, best_index]),
        }
        examples.append(example)
        split_counts[condition["split"]] += 1

    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / "targets.npz"
    _atomic_npz(
        arrays_path,
        profiles=profiles,
        scores=scores,
        candidate_mask=candidate_mask,
    )
    payload = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "campaign_id": campaign["campaign_id"],
        "ranking": ranking,
        "profile_bins": int(profile_bins),
        "candidate_names": list(candidates),
        "accepted_candidate_count": int(candidate_mask.sum()),
        "rejected_candidate_count": int(candidate_mask.size - candidate_mask.sum()),
        "example_count": len(examples),
        "split_counts": dict(sorted(split_counts.items())),
        "source_hashes": {
            "manifest": _sha256_file(manifest_path),
            "campaign": _sha256_file(campaign_path),
            "report": _sha256_file(campaign_output / "report.json"),
            "labels": _sha256_file(campaign_output / "labels.json"),
        },
        "arrays": arrays_path.name,
        "examples": examples,
    }
    _atomic_json(destination / "dataset.json", payload)
    return payload


def merge_distillation_datasets(
    base_dataset_path,
    augmentation_dataset_path,
    destination,
    *,
    augmentation_splits=("train",),
):
    """Merge training augmentation while preserving the base evaluation splits.

    The candidate target space must be identical in both inputs.  By default only
    training rows are admitted from the augmentation dataset, which keeps the
    original validation and test sets frozen for an unbiased retraining comparison.
    """
    base_path = Path(base_dataset_path)
    augmentation_path = Path(augmentation_dataset_path)
    destination = Path(destination)
    requested_splits = tuple(augmentation_splits)
    if not requested_splits or any(not isinstance(split, str) for split in requested_splits):
        raise ValueError("At least one augmentation split name is required")
    if len(set(requested_splits)) != len(requested_splits):
        raise ValueError("Augmentation split names must be unique")

    metadata = [json.loads(path.read_text()) for path in (base_path, augmentation_path)]
    base, augmentation = metadata
    for field in ("profile_bins", "candidate_names", "ranking"):
        if base.get(field) != augmentation.get(field):
            raise ValueError(f"Distillation datasets differ in {field}")

    loaded = []
    try:
        for path, item in zip((base_path, augmentation_path), metadata, strict=True):
            arrays = np.load(path.parent / item["arrays"])
            values = {
                "profiles": arrays["profiles"].copy(),
                "scores": arrays["scores"].copy(),
                "candidate_mask": (
                    arrays["candidate_mask"].copy()
                    if "candidate_mask" in arrays
                    else np.ones_like(arrays["scores"], dtype=bool)
                ),
            }
            loaded.append(values)
            arrays.close()
            count = len(item["examples"])
            if any(value.shape[0] != count for value in values.values()):
                raise ValueError(f"Dataset arrays do not match metadata: {path}")
    except (KeyError, OSError) as error:
        raise ValueError("Invalid distillation dataset arrays") from error

    augmentation_indices = [
        index
        for index, example in enumerate(augmentation["examples"])
        if example["split"] in requested_splits
    ]
    if not augmentation_indices:
        raise ValueError("No augmentation examples matched the requested splits")
    examples = list(base["examples"]) + [
        augmentation["examples"][index] for index in augmentation_indices
    ]
    sample_ids = [example["sample_id"] for example in examples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Merged datasets contain duplicate sample IDs")

    arrays = {
        name: np.concatenate((loaded[0][name], loaded[1][name][augmentation_indices]), axis=0)
        for name in ("profiles", "scores", "candidate_mask")
    }
    split_counts = Counter(example["split"] for example in examples)
    source_hashes = {
        "base_dataset": _sha256_file(base_path),
        "base_arrays": _sha256_file(base_path.parent / base["arrays"]),
        "augmentation_dataset": _sha256_file(augmentation_path),
        "augmentation_arrays": _sha256_file(
            augmentation_path.parent / augmentation["arrays"]
        ),
    }
    identity = {
        "source_hashes": source_hashes,
        "augmentation_splits": list(requested_splits),
    }
    dataset_id = "merged_distillation_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / "targets.npz"
    _atomic_npz(arrays_path, **arrays)
    payload = {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "campaign_ids": [base.get("campaign_id"), augmentation.get("campaign_id")],
        "ranking": base["ranking"],
        "profile_bins": base["profile_bins"],
        "candidate_names": base["candidate_names"],
        "accepted_candidate_count": int(arrays["candidate_mask"].sum()),
        "rejected_candidate_count": int(arrays["candidate_mask"].size - arrays["candidate_mask"].sum()),
        "example_count": len(examples),
        "split_counts": dict(sorted(split_counts.items())),
        "augmentation_splits": list(requested_splits),
        "source_hashes": source_hashes,
        "arrays": arrays_path.name,
        "examples": examples,
    }
    _atomic_json(destination / "dataset.json", payload)
    return payload
