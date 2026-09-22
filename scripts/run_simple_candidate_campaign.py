#!/usr/bin/env python3
"""Restartable multi-GPU candidate physics and Pareto-label campaign."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
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
    parser.add_argument("--shard-strategy", choices=("index", "hash"), default="index")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--source-revision",
        help="Git revision used to reconstruct a campaign's launch-time source fingerprint",
    )
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
    for variant in campaign.get("candidate_variants", []):
        name = variant["candidate"]
        base_name = variant["base_candidate"]
        factor = variant["cell_factor"]
        if name in candidate_map:
            raise ValueError(f"Duplicate mesh candidate: {name}")
        if base_name not in candidate_map:
            raise ValueError(f"Unknown base mesh candidate: {base_name}")
        if not np.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("Candidate cell_factor must be in (0, 1]")
        candidate_map[name] = dict(
            candidate_map[base_name],
            candidate=name,
            base_candidate=base_name,
            cell_factor=float(factor),
        )
    if not set(campaign["candidate_names"]) <= candidate_map.keys():
        raise ValueError("Campaign contains an unknown mesh candidate")
    ranking = campaign.get("ranking", {"mode": "matched_cell_updates"})
    if ranking["mode"] not in {"matched_cell_updates", "fixed_axis_soft_nt"}:
        raise ValueError(f"Unsupported candidate ranking mode: {ranking['mode']}")
    if ranking["mode"] == "fixed_axis_soft_nt":
        exponent = ranking.get("nt_cost_exponent")
        if not np.isfinite(exponent) or exponent < 0:
            raise ValueError("fixed_axis_soft_nt requires a nonnegative nt_cost_exponent")
        scaled = [
            name
            for name in campaign["candidate_names"]
            if candidate_map[name].get("cell_factor", 1.0) != 1.0
        ]
        if scaled:
            raise ValueError(f"Exact-axis ranking cannot use scaled candidates: {scaled}")
    return manifest, campaign, geometries, conditions, candidate_map


def source_hashes_at_revision(revision):
    """Reconstruct the broad legacy source fingerprint at a Git revision."""
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/scattermesh"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    if not paths:
        raise ValueError(f"No scattermesh sources found at revision {revision}")
    result = {}
    for path in sorted(path for path in paths if path.endswith(".py")):
        content = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
        result[path] = hashlib.sha256(content).hexdigest()
    result["pilot_mesh_numerical_schema"] = str(PILOT.NUMERICAL_SCHEMA)
    return result


def campaign_source_hashes(output, source_revision=None):
    """Pin one source fingerprint per output cache and reuse it on every resume."""
    path = output / "source_hashes.json"
    if path.exists():
        payload = json.loads(path.read_text())
        return payload["hashes"]
    hashes = (
        source_hashes_at_revision(source_revision) if source_revision else PILOT.source_hashes()
    )
    PILOT.atomic_json(
        path,
        {
            "schema_version": 1,
            "source_revision": source_revision,
            "hashes": hashes,
        },
    )
    return hashes


def select_shard(cases, shard, shards, strategy="index"):
    """Assign complete logical cases deterministically without candidate-order bias."""
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Require 0 <= shard < shards")
    if strategy == "index":
        return [case for index, case in enumerate(cases) if index % shards == shard]
    if strategy == "hash":
        return [
            case
            for case in cases
            if int.from_bytes(hashlib.sha256(case["case_id"].encode()).digest()[:8], "big")
            % shards
            == shard
        ]
    raise ValueError(f"Unsupported shard strategy: {strategy}")


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
        duration_schedule = campaign.get("duration_schedule_s")
        if duration_schedule is not None:
            if (
                not duration_schedule
                or any(not np.isfinite(value) or value <= 0 for value in duration_schedule)
                or list(duration_schedule) != sorted(set(duration_schedule))
            ):
                raise ValueError("duration_schedule_s must be strictly increasing and positive")
            initial_duration = duration_schedule[0]
        else:
            initial_duration = campaign["duration_s"]
        for candidate_name in campaign["candidate_names"]:
            candidate = candidate_map[candidate_name]
            target_cells = condition["cells_x"]
            cell_factor = candidate.get("cell_factor")
            candidate_cells = (
                max(4, int(np.floor(target_cells * cell_factor)))
                if cell_factor is not None
                else target_cells
            )
            config = PILOT.definitions(
                [scene],
                [candidate_cells],
                [candidate],
                duration=initial_duration,
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
            if cell_factor is not None:
                config.update(
                    target_cells_x=condition["cells_x"],
                    target_cells_y=condition["cells_y"],
                    candidate_cell_factor=cell_factor,
                )
            if duration_schedule is not None:
                config["duration_schedule_s"] = list(duration_schedule)
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


def duration_attempt_configs(config):
    """Expand a logical case into deterministic duration attempts."""
    schedule = config.get("duration_schedule_s")
    if schedule is None:
        return [config]
    return [dict(config, duration=duration, duration_attempt_index=index) for index, duration in enumerate(schedule)]


def archive_attempt(output, case_id, attempt_index):
    """Retain a failed attempt before the logical case is overwritten."""
    source = output / "cases" / case_id
    destination = output / "attempts" / case_id / f"attempt_{attempt_index:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("record.json", "spectra.npz"):
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)


def run_campaign_case(config, output, device, sources):
    """Run or resume one logical case through its declared duration schedule."""
    attempts = duration_attempt_configs(config)
    if len(attempts) == 1:
        return PILOT.run_case(config, output, device, sources)
    directory = output / "cases" / config["case_id"]
    record_path, arrays_path = directory / "record.json", directory / "spectra.npz"
    fingerprints = [PILOT.sha256_json(dict(config=row, sources=sources)) for row in attempts]
    current_index = None
    try:
        current = json.loads(record_path.read_text())
        current_index = fingerprints.index(current.get("fingerprint"))
        current = PILOT.valid_cache(record_path, arrays_path, fingerprints[current_index])
        if current is None:
            current_index = None
    except (OSError, ValueError, json.JSONDecodeError):
        current = None
        current_index = None
    if current is not None:
        if current["accepted"] or current_index == len(attempts) - 1:
            return current, True
        archive_attempt(output, config["case_id"], current_index)
    start = 0 if current_index is None else current_index + 1
    for index in range(start, len(attempts)):
        record, _ = PILOT.run_case(attempts[index], output, device, sources)
        if record["accepted"] or index == len(attempts) - 1:
            return record, False
        archive_attempt(output, config["case_id"], index)
    raise RuntimeError("Duration schedule exhausted without a terminal record")


def load_record(config, output, sources, campaign_id):
    migrate_legacy_record(config, output, sources, campaign_id)
    directory = output / "cases" / config["case_id"]
    for attempt in duration_attempt_configs(config):
        fingerprint = PILOT.sha256_json(dict(config=attempt, sources=sources))
        record = PILOT.valid_cache(
            directory / "record.json", directory / "spectra.npz", fingerprint
        )
        if record is not None:
            return record
    raise ValueError(f"Missing or stale campaign case: {config['case_id']}")


def label_diversity(labels, *, meaningful_improvement=1.05):
    """Summarize whether Pareto labels contain learnable, split-safe variation."""
    by_split = defaultdict(list)
    for illumination_id, label in labels.items():
        by_split[label["split"]].append((illumination_id, label))

    result = {}
    for split, groups in sorted(by_split.items()):
        winners = []
        for illumination_id, label in groups:
            for budget, row in label["budgets"].items():
                if (
                    row["best_case"] != row["uniform_case"]
                    and row["improvement"] >= meaningful_improvement
                ):
                    winners.append(
                        dict(
                            illumination_id=illumination_id,
                            lineage_id=label["lineage_id"],
                            budget=int(budget),
                            candidate=row["best_candidate"],
                            improvement=row["improvement"],
                        )
                    )
        improvements = sorted(row["improvement"] for row in winners)
        result[split] = dict(
            illumination_groups=len(groups),
            budget_labels=sum(len(label["budgets"]) for _, label in groups),
            meaningful_nonuniform_wins=len(winners),
            winning_lineages=sorted({row["lineage_id"] for row in winners}),
            winning_candidates=dict(sorted(Counter(row["candidate"] for row in winners).items())),
            winning_budgets=dict(sorted(Counter(str(row["budget"]) for row in winners).items())),
            median_improvement=(float(np.median(improvements)) if improvements else 1.0),
            maximum_improvement=max(improvements, default=1.0),
        )
    return result


def training_readiness(diversity, campaign_complete):
    """Apply the conservative pre-M5 label gate documented in the full report."""
    train = diversity.get("train", {})
    validation = diversity.get("validation", {})
    test = diversity.get("test", {})
    checks = dict(
        campaign_complete=campaign_complete,
        train_has_multiple_winning_lineages=len(train.get("winning_lineages", ())) >= 2,
        train_has_multiple_winning_candidates=len(train.get("winning_candidates", {})) >= 2,
        train_has_multiple_winning_budgets=len(train.get("winning_budgets", {})) >= 2,
        validation_has_headroom=validation.get("meaningful_nonuniform_wins", 0) >= 1,
        test_has_headroom=test.get("meaningful_nonuniform_wins", 0) >= 1,
    )
    return dict(
        decision="ready_for_m5_pilot" if all(checks.values()) else "not_ready_for_m5",
        meaningful_improvement_threshold=1.05,
        checks=checks,
    )


def ranked_candidates(rows, baseline, budget, ranking):
    """Return accepted candidates and comparable scores for one budget label."""
    mode = ranking.get("mode", "matched_cell_updates")
    accepted = [row for row in rows if row["accepted"]]
    if mode == "matched_cell_updates":
        feasible = [row for row in accepted if row["cell_updates"] <= baseline["cell_updates"]]
        return [(row, row["joint_scattering_loss"]) for row in feasible]
    if mode == "fixed_axis_soft_nt":
        exponent = ranking["nt_cost_exponent"]
        exact = [
            row
            for row in accepted
            if row["config"].get("target_cells_x", row["config"]["cells"]) == budget
            and row["config"]["cells"] == budget
            and row["Nx"] == baseline["Nx"]
            and row["Ny"] == baseline["Ny"]
        ]
        return [
            (
                row,
                row["joint_scattering_loss"]
                * (row["Nt"] / baseline["Nt"]) ** exponent,
            )
            for row in exact
        ]
    raise ValueError(f"Unsupported candidate ranking mode: {mode}")


def candidate_scorecard(groups, ranking=None, *, meaningful_improvement=1.05):
    """Compare each policy family with every affordable uniform update cap."""
    ranking = {"mode": "matched_cell_updates"} if ranking is None else ranking
    status_counts = defaultdict(Counter)
    improvements = defaultdict(list)
    for rows in groups.values():
        for row in rows:
            status_counts[row["config"]["candidate"]][row["status"]] += 1
        accepted = [row for row in rows if row["accepted"]]
        uniform = {
            row["config"].get("target_cells_x", row["config"]["cells"]): row
            for row in accepted
            if row["config"]["candidate"] == "uniform"
        }
        candidate_names = {
            row["config"]["candidate"]
            for row in rows
            if row["config"]["candidate"] != "uniform"
        }
        for budget, baseline in uniform.items():
            for candidate_name in candidate_names:
                ranked = [
                    (row, score)
                    for row, score in ranked_candidates(rows, baseline, budget, ranking)
                    if row["config"]["candidate"] == candidate_name
                ]
                if ranked:
                    _, score = min(ranked, key=lambda item: item[1])
                    improvements[candidate_name].append(
                        baseline["joint_scattering_loss"] / score
                    )
    result = {}
    for candidate_name in sorted(status_counts):
        values = improvements[candidate_name]
        result[candidate_name] = {
            "status_counts": dict(sorted(status_counts[candidate_name].items())),
            "affordable_comparisons": len(values),
            "meaningful_wins": sum(value >= meaningful_improvement for value in values),
            "median_improvement": float(np.median(values)) if values else None,
            "maximum_improvement": max(values, default=None),
        }
    return result


def summarize(cases, output, sources, campaign):
    records = [load_record(config, output, sources, campaign["campaign_id"]) for config in cases]
    groups = defaultdict(list)
    for record in records:
        groups[record["config"]["illumination_id"]].append(record)
    labels, invalid_groups = {}, []
    for illumination_id, rows in groups.items():
        accepted = [row for row in rows if row["accepted"]]
        uniform = {
            row["config"].get("target_cells_x", row["config"]["cells"]): row
            for row in accepted
            if row["config"]["candidate"] == "uniform"
        }
        expected_budgets = sorted(
            {
                row["config"].get("target_cells_x", row["config"]["cells"])
                for row in rows
            }
        )
        if set(uniform) != set(expected_budgets):
            invalid_groups.append(illumination_id)
        budget_labels = {}
        for cells, baseline in sorted(uniform.items()):
            ranked = ranked_candidates(
                rows,
                baseline,
                cells,
                campaign.get("ranking", {"mode": "matched_cell_updates"}),
            )
            best, best_score = min(ranked, key=lambda item: item[1])
            budget_labels[str(cells)] = dict(
                update_cap=baseline["cell_updates"],
                uniform_case=baseline["case_id"],
                uniform_loss=baseline["joint_scattering_loss"],
                best_case=best["case_id"],
                best_candidate=best["config"]["candidate"],
                best_cells=best["config"]["cells"],
                best_cell_factor=best["config"].get("candidate_cell_factor", 1.0),
                best_updates=best["cell_updates"],
                best_nt=best["Nt"],
                best_loss=best["joint_scattering_loss"],
                raw_accuracy_improvement=(
                    baseline["joint_scattering_loss"] / best["joint_scattering_loss"]
                ),
                update_cost_ratio=best["cell_updates"] / baseline["cell_updates"],
                selection_score=best_score,
                improvement=baseline["joint_scattering_loss"] / best_score,
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
    diversity = label_diversity(labels)
    campaign_complete = not invalid_groups
    report = dict(
        schema_version=2,
        dataset_id=cases[0]["dataset_id"],
        campaign_id=campaign["campaign_id"],
        decision="accepted" if campaign_complete else "incomplete_uniform_baselines",
        case_count=len(records),
        status_counts=dict(sorted(statuses.items())),
        illumination_group_count=len(groups),
        invalid_uniform_groups=invalid_groups,
        nonuniform_budget_win_count=sum(value > 1 for value in improvements),
        maximum_improvement=max(improvements, default=1.0),
        ranking=campaign.get("ranking", {"mode": "matched_cell_updates"}),
        label_diversity=diversity,
        candidate_scorecard=candidate_scorecard(groups, campaign.get("ranking")),
        training_readiness=training_readiness(diversity, campaign_complete),
        labels_path=str(output / "labels.json"),
    )
    PILOT.atomic_json(output / "labels.json", dict(schema_version=2, groups=labels))
    PILOT.atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    args = parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    inputs = load_inputs(args.manifest, args.campaign)
    cases = case_definitions(*inputs)
    sources = campaign_source_hashes(args.output, args.source_revision)
    if args.summarize:
        summarize(cases, args.output, sources, inputs[1])
        return
    selected = select_shard(cases, args.shard, args.shards, args.shard_strategy)
    for index, config in enumerate(selected, 1):
        migrate_legacy_record(config, args.output, sources, inputs[1]["campaign_id"])
        record, cached = run_campaign_case(config, args.output, args.device, sources)
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
