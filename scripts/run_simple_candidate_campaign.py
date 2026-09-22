#!/usr/bin/env python3
"""Restartable multi-GPU candidate physics and Pareto-label campaign."""

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "scripts/pilot_mesh_headroom.py"
SPEC = importlib.util.spec_from_file_location("scattermesh_headroom_pilot", PILOT_PATH)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/simple_dielectric_pool.json")
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/simple_candidate_pilot.json")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/simple_candidate_pilot"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def load_inputs(manifest_path, campaign_path):
    manifest = json.loads(manifest_path.read_text())
    campaign = json.loads(campaign_path.read_text())
    if campaign["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Campaign dataset_id does not match the scene manifest")
    geometries = {row["geometry_id"]: row for row in manifest["geometries"]}
    conditions = {row["task_id"]: row for row in manifest["conditions"]}
    if (
        len(geometries) != manifest["geometry_count"]
        or len(conditions) != manifest["condition_count"]
    ):
        raise ValueError("Manifest IDs are not unique")
    if not set(campaign["geometry_ids"]) <= geometries.keys():
        raise ValueError("Campaign contains an unknown geometry")
    if not set(campaign["condition_ids"]) <= conditions.keys():
        raise ValueError("Campaign contains an unknown condition")
    candidate_map = {row["candidate"]: row for row in PILOT.CANDIDATES}
    if not set(campaign["candidate_names"]) <= candidate_map.keys():
        raise ValueError("Campaign contains an unknown mesh candidate")
    return manifest, campaign, geometries, conditions, candidate_map


def case_definitions(manifest, campaign, geometries, conditions, candidate_map):
    cases = []
    for condition_id in campaign["condition_ids"]:
        condition = conditions[condition_id]
        geometry = geometries[condition["geometry_id"]]
        scene = dict(
            scene_id=condition_id,
            radius=geometry["radius_m"],
            center=geometry["center_m"],
            angle=condition["incidence_angle_rad"],
            material=dict(
                epsilon_r=geometry["epsilon_r"],
                sigma_e=geometry["sigma_e_s_per_m"],
            ),
        )
        for candidate_name in campaign["candidate_names"]:
            config = PILOT.definitions(
                [scene],
                [condition["cells_x"]],
                [candidate_map[candidate_name]],
                duration=campaign["duration_s"],
                pec_mode=None,
            )[0]
            config.update(
                dataset_id=manifest["dataset_id"],
                condition_id=condition_id,
                geometry_id=condition["geometry_id"],
                lineage_id=condition["lineage_id"],
                illumination_id=condition["illumination_id"],
                split=condition["split"],
                family=condition["family"],
            )
            config["case_id"] = f"{condition_id}_{candidate_name}"
            cases.append(config)
    return cases


def migrate_legacy_record(config, output, sources, campaign_id):
    """Remove nonnumerical campaign membership from an otherwise exact cached case."""
    directory = output / "cases" / config["case_id"]
    record_path, arrays_path = directory / "record.json", directory / "spectra.npz"
    if not record_path.exists() or not arrays_path.exists():
        return
    try:
        record = json.loads(record_path.read_text())
        legacy = dict(config, campaign_id=campaign_id)
        legacy_fingerprint = PILOT.sha256_json(dict(config=legacy, sources=sources))
        if record.get("config") != legacy or record.get("fingerprint") != legacy_fingerprint:
            return
        with np.load(arrays_path) as arrays:
            if not (
                np.isfinite(arrays["complex_far_field"]).all()
                and np.isfinite(arrays["analytic_complex_far_field"]).all()
            ):
                return
        record["config"] = config
        record["fingerprint"] = PILOT.sha256_json(dict(config=config, sources=sources))
        PILOT.atomic_json(record_path, record)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return


def load_record(config, output, sources, campaign_id):
    migrate_legacy_record(config, output, sources, campaign_id)
    fingerprint = PILOT.sha256_json(dict(config=config, sources=sources))
    directory = output / "cases" / config["case_id"]
    record = PILOT.valid_cache(directory / "record.json", directory / "spectra.npz", fingerprint)
    if record is None:
        raise ValueError(f"Missing or stale campaign case: {config['case_id']}")
    return record


def summarize(cases, output, sources, campaign):
    records = [load_record(config, output, sources, campaign["campaign_id"]) for config in cases]
    groups = defaultdict(list)
    for record in records:
        groups[record["config"]["illumination_id"]].append(record)
    labels, invalid_groups = {}, []
    for illumination_id, rows in groups.items():
        accepted = [row for row in rows if row["accepted"]]
        uniform = {
            row["config"]["cells"]: row
            for row in accepted
            if row["config"]["candidate"] == "uniform"
        }
        expected_budgets = sorted({row["config"]["cells"] for row in rows})
        if set(uniform) != set(expected_budgets):
            invalid_groups.append(illumination_id)
        budget_labels = {}
        for cells, baseline in sorted(uniform.items()):
            feasible = [row for row in accepted if row["cell_updates"] <= baseline["cell_updates"]]
            best = min(feasible, key=lambda row: row["joint_scattering_loss"])
            budget_labels[str(cells)] = dict(
                update_cap=baseline["cell_updates"],
                uniform_case=baseline["case_id"],
                uniform_loss=baseline["joint_scattering_loss"],
                best_case=best["case_id"],
                best_loss=best["joint_scattering_loss"],
                improvement=baseline["joint_scattering_loss"] / best["joint_scattering_loss"],
            )
        first = rows[0]["config"]
        labels[illumination_id] = dict(
            geometry_id=first["geometry_id"],
            lineage_id=first["lineage_id"],
            split=first["split"],
            family=first["family"],
            pareto_cases=PILOT.pareto(accepted),
            budgets=budget_labels,
        )
    statuses = Counter(row["status"] for row in records)
    improvements = [
        budget["improvement"] for label in labels.values() for budget in label["budgets"].values()
    ]
    report = dict(
        schema_version=1,
        dataset_id=cases[0]["dataset_id"],
        campaign_id=campaign["campaign_id"],
        decision="accepted" if not invalid_groups else "incomplete_uniform_baselines",
        case_count=len(records),
        status_counts=dict(sorted(statuses.items())),
        illumination_group_count=len(groups),
        invalid_uniform_groups=invalid_groups,
        nonuniform_budget_win_count=sum(value > 1 for value in improvements),
        maximum_improvement=max(improvements, default=1.0),
        labels_path=str(output / "labels.json"),
    )
    PILOT.atomic_json(output / "labels.json", dict(schema_version=1, groups=labels))
    PILOT.atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    args = parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    inputs = load_inputs(args.manifest, args.campaign)
    cases = case_definitions(*inputs)
    sources = PILOT.source_hashes()
    if args.summarize:
        summarize(cases, args.output, sources, inputs[1])
        return
    selected = [case for index, case in enumerate(cases) if index % args.shards == args.shard]
    for index, config in enumerate(selected, 1):
        migrate_legacy_record(config, args.output, sources, inputs[1]["campaign_id"])
        record, cached = PILOT.run_case(config, args.output, args.device, sources)
        outcome = (
            f"loss={record['joint_scattering_loss']:.6g} updates={record['cell_updates']}"
            if record["status"] != "mesh_infeasible"
            else f"infeasible={record['reason']}"
        )
        print(
            f"[{index}/{len(selected)}] {config['case_id']} {outcome} "
            f"{'cached' if cached else 'ran'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
