#!/usr/bin/env python3
"""Merge qualified pair pilots with historical single-circle distillation labels."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from scattermesh.distillation import axis_probability
from scattermesh.sparse import sparse_cluster_metrics


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _split(scene_id):
    """Freeze scene-level splits before inspecting sparse model results."""
    validation = {
        "wide_vertical_northwest",
        "mixed_circle_rectangle_moderate",
    }
    test = {
        "close_size_contrast_east",
        "moderate_equal_upper_center",
        "mixed_rectangle_rectangle_close",
        "pec_rectangle_rectangle_wide",
    }
    return "validation" if scene_id in validation else "test" if scene_id in test else "train"


def build(base_path, pilots, output):
    base_path = Path(base_path)
    output = Path(output)
    base = json.loads(base_path.read_text())
    if base["ranking"] != {"mode": "fixed_axis_soft_nt", "nt_cost_exponent": 0.1}:
        raise ValueError("Historical candidate ranking differs from sparse pilots")
    with np.load(base_path.parent / base["arrays"]) as arrays:
        base_profiles = arrays["profiles"].copy()
        base_scores = arrays["scores"].copy()
        base_mask = arrays["candidate_mask"].copy()
    if base_profiles.shape[0] != len(base["examples"]):
        raise ValueError("Historical arrays and examples do not match")
    pilot_data = []
    names = list(base["candidate_names"])
    for config_path, directory in pilots:
        config_path, directory = Path(config_path), Path(directory)
        config = json.loads(config_path.read_text())
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text())
        if report["decision"] != "passes_sparse_pair_headroom" or not all(
            report["checks"].values()
        ):
            raise ValueError(f"Sparse pilot did not pass its predeclared gate: {directory}")
        for policy in config["policies"]:
            if policy["name"] not in names:
                names.append(policy["name"])
        pilot_data.append((config_path, directory, config, report))

    count = len(base["examples"]) + sum(
        len(config["scenes"]) * len(config["budgets"])
        for _, _, config, _ in pilot_data
    )
    bins = base["profile_bins"]
    profiles = np.full((count, len(names), 2, bins), 1 / bins, dtype=np.float32)
    scores = np.ones((count, len(names)), dtype=np.float64)
    mask = np.zeros((count, len(names)), dtype=bool)
    old = len(base["examples"])
    for source_index, name in enumerate(base["candidate_names"]):
        target_index = names.index(name)
        profiles[:old, target_index] = base_profiles[:, source_index]
        scores[:old, target_index] = base_scores[:, source_index]
        mask[:old, target_index] = base_mask[:, source_index]
    examples = []
    for item in base["examples"]:
        updated = dict(item)
        updated["best_candidate_index"] = names.index(item["best_candidate"])
        examples.append(updated)

    index = old
    for _, directory, config, _ in pilot_data:
        for scene in config["scenes"]:
            metrics = sparse_cluster_metrics(scene, domain=config["domain_m"])
            for cells in config["budgets"]:
                records = {}
                for policy in config["policies"]:
                    case_id = f"candidate_{scene['scene_id']}_n{cells}_{policy['name']}"
                    case_dir = directory / "candidates" / case_id
                    record = json.loads((case_dir / "record.json").read_text())
                    if record["definition"]["cells"] != cells:
                        raise ValueError(f"Sparse candidate has wrong budget: {case_id}")
                    records[policy["name"]] = record
                uniform = records["uniform"]
                if not uniform["accepted"]:
                    raise ValueError("Sparse condition lacks a settled uniform baseline")
                for policy in config["policies"]:
                    name = policy["name"]
                    record = records[name]
                    candidate_index = names.index(name)
                    if not record["accepted"]:
                        continue
                    with np.load(directory / "candidates" / record["case_id"] / "spectra.npz") as arrays:
                        x, y = arrays["x"].copy(), arrays["y"].copy()
                    if len(x) != cells + 1 or len(y) != cells + 1:
                        raise ValueError(f"Sparse candidate has wrong axis length: {record['case_id']}")
                    profiles[index, candidate_index, 0] = axis_probability(x, bins)
                    profiles[index, candidate_index, 1] = axis_probability(y, bins)
                    scores[index, candidate_index] = record["joint_scattering_loss"] * (
                        record["Nt"] / uniform["Nt"]
                    ) ** config["ranking"]["nt_cost_exponent"]
                    mask[index, candidate_index] = True
                best_index = int(np.argmin(np.where(mask[index], scores[index], np.inf)))
                split = scene.get("split", _split(scene["scene_id"]))
                examples.append(
                    {
                        "sample_id": f"sparse_{scene['scene_id']}_n{cells}",
                        "geometry_id": scene["scene_id"],
                        "lineage_id": scene["scene_id"],
                        "split": split,
                        "family": "sparse_pair",
                        "objects": scene["objects"],
                        "shape_topology": metrics["shape_topology"],
                        "material_topology": metrics["material_topology"],
                        "gap_stratum": scene["gap_stratum"],
                        "feature_size_m": min(
                            metrics["minimum_gap_m"],
                            *(
                                2 * obj["radius_m"] if obj["shape"] == "circle"
                                else min(
                                    obj["bounds_m"][1] - obj["bounds_m"][0],
                                    obj["bounds_m"][3] - obj["bounds_m"][2],
                                )
                                for obj in scene["objects"]
                            ),
                        ),
                        "incidence_angle_rad": scene["incidence_angle_rad"],
                        "frequencies_hz": config["frequencies_hz"],
                        "cells_x": cells,
                        "cells_y": cells,
                        "best_candidate": names[best_index],
                        "best_candidate_index": best_index,
                        "uniform_score": float(scores[index, names.index("uniform")]),
                        "best_score": float(scores[index, best_index]),
                    }
                )
                index += 1
    if index != count or len(examples) != count:
        raise ValueError("Merged dataset size mismatch")
    ids = [item["sample_id"] for item in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs in merged dataset")
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "targets.npz"
    _atomic_npz(arrays_path, profiles=profiles, scores=scores, candidate_mask=mask)
    payload = {
        "schema_version": 2,
        "dataset_id": "sparse_joint_" + hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()[:16],
        "ranking": base["ranking"],
        "profile_bins": bins,
        "candidate_names": names,
        "example_count": count,
        "split_counts": dict(Counter(item["split"] for item in examples)),
        "family_split_counts": {
            family: dict(Counter(item["split"] for item in examples if item["family"] == family))
            for family in sorted({item["family"] for item in examples})
        },
        "source_hashes": {
            "base_dataset": _sha256_file(base_path),
            "base_arrays": _sha256_file(base_path.parent / base["arrays"]),
            **{
                f"pilot_{index}_config": _sha256_file(config_path)
                for index, (config_path, _, _, _) in enumerate(pilot_data)
            },
            **{
                f"pilot_{index}_report": _sha256_file(directory / "report.json")
                for index, (_, directory, _, _) in enumerate(pilot_data)
            },
        },
        "arrays": arrays_path.name,
        "examples": examples,
    }
    _atomic_json(output / "dataset.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--pilot", nargs=2, action="append", metavar=("CONFIG", "OUTPUT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.base_dataset, args.pilot, args.output)
    print(json.dumps({key: payload[key] for key in ("dataset_id", "example_count", "split_counts", "family_split_counts")}, indent=2))


if __name__ == "__main__":
    main()
