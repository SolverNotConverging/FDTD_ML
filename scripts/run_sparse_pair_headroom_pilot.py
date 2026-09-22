#!/usr/bin/env python3
"""Restartable exact-budget headroom pilot for localized dielectric-circle pairs."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from scattermesh import (
    Grid,
    PlaneWave,
    simulate_cuda,
    sparse_candidate_axes,
    sparse_cluster_metrics,
    sparse_scene_objects,
)
from scattermesh.generalization import widest_non_pml_monitor_bounds
from scattermesh.metrics import scattering_loss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/sparse_pair_headroom_pilot.json"
NUMERICAL_SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("runs/sparse_pair_headroom_pilot"))
    parser.add_argument("--phase", choices=("references", "candidates", "summarize"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--reference-sources", type=Path)
    parser.add_argument("--candidate-sources", type=Path)
    return parser.parse_args()


def _sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def source_hashes(config_path):
    paths = sorted((ROOT / "src/scattermesh").glob("*.py"))
    return {
        **{str(path.relative_to(ROOT)): _sha256_file(path) for path in paths},
        "config": _sha256_file(config_path),
        "sparse_pair_numerical_schema": str(NUMERICAL_SCHEMA),
    }


def load_config(path):
    config = json.loads(path.read_text())
    if config.get("ranking", {}).get("mode") != "fixed_axis_soft_nt":
        raise ValueError("Sparse headroom pilot requires exact-axis soft-Nt ranking")
    if len(config["reference_budgets"]) < 2:
        raise ValueError("At least two reference resolutions are required")
    names = [policy["name"] for policy in config["policies"]]
    if len(set(names)) != len(names) or "uniform" not in names:
        raise ValueError("Candidate policy names must be unique and include uniform")
    for scene in config["scenes"]:
        metrics = sparse_cluster_metrics(scene, domain=config["domain_m"])
        if not 0.005 <= metrics["occupied_area_fraction"] <= 0.12:
            raise ValueError(f"Scene outside pilot occupancy stratum: {scene['scene_id']}")
        if max(
            metrics["projected_x_support_fraction"], metrics["projected_y_support_fraction"]
        ) > 0.45:
            raise ValueError(f"Scene outside pilot projected-support stratum: {scene['scene_id']}")
    return config


def definitions(config, phase):
    if phase == "references":
        return [
            {
                "case_id": f"reference_{scene['scene_id']}_n{cells}",
                "phase": phase,
                "scene": scene,
                "cells": cells,
                "policy": {"name": "uniform", "kind": "uniform", "max_ratio": 1.0},
            }
            for scene in config["scenes"]
            for cells in config["reference_budgets"]
        ]
    if phase == "candidates":
        return [
            {
                "case_id": f"candidate_{scene['scene_id']}_n{cells}_{policy['name']}",
                "phase": phase,
                "scene": scene,
                "cells": cells,
                "policy": policy,
            }
            for scene in config["scenes"]
            for cells in config["budgets"]
            for policy in config["policies"]
        ]
    raise ValueError(f"No definitions for phase {phase}")


def _case_paths(output, definition):
    directory = output / definition["phase"] / definition["case_id"]
    return directory / "record.json", directory / "spectra.npz"


def _valid_cache(record_path, arrays_path, fingerprint, angle_count):
    if not record_path.exists() or not arrays_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text())
        with np.load(arrays_path) as arrays:
            valid = (
                record.get("fingerprint") == fingerprint
                and record.get("status") in {"accepted", "hard_limit_unsettled"}
                and arrays["complex_far_field"].shape[1] == angle_count
                and np.isfinite(arrays["complex_far_field"]).all()
            )
        return record if valid else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _reference_fingerprint(output, config, scene):
    cells = max(config["reference_budgets"])
    definition = next(
        row
        for row in definitions(config, "references")
        if row["scene"]["scene_id"] == scene["scene_id"] and row["cells"] == cells
    )
    record_path, arrays_path = _case_paths(output, definition)
    if not record_path.exists() or not arrays_path.exists():
        raise ValueError(f"Fine reference is missing for {scene['scene_id']}")
    record = json.loads(record_path.read_text())
    if record.get("status") != "accepted":
        raise ValueError(f"Fine reference is not settled for {scene['scene_id']}")
    return record["fingerprint"], arrays_path


def make_grid(config, definition):
    policy = definition["policy"]
    x, y = sparse_candidate_axes(
        config["domain_m"], definition["cells"], definition["scene"], policy
    )
    return Grid(x, y, max_ratio=policy["max_ratio"])


def run_case(config, definition, output, device, sources):
    reference_identity = None
    reference_path = None
    if definition["phase"] == "candidates":
        reference_identity, reference_path = _reference_fingerprint(
            output, config, definition["scene"]
        )
    fingerprint = _sha256_json(
        {
            "definition": definition,
            "protocol": {
                key: config[key]
                for key in (
                    "domain_m",
                    "frequencies_hz",
                    "far_field_angle_count",
                    "duration_schedule_s",
                    "pml_thickness_m",
                    "tail_limit",
                    "material_samples_per_axis",
                )
            },
            "sources": sources,
            "reference": reference_identity,
        }
    )
    record_path, arrays_path = _case_paths(output, definition)
    cached = _valid_cache(
        record_path, arrays_path, fingerprint, config["far_field_angle_count"]
    )
    if cached is not None:
        return cached, True

    grid = make_grid(config, definition)
    objects = sparse_scene_objects(definition["scene"])
    monitor = widest_non_pml_monitor_bounds(
        grid, [obj.bounds for obj in objects], config["pml_thickness_m"]
    )
    source = PlaneWave(
        1e9,
        1e-9,
        9e-9,
        angle=definition["scene"]["incidence_angle_rad"],
        origin=(config["domain_m"] / 2, config["domain_m"] / 2),
    )
    angles = np.linspace(0, 2 * np.pi, config["far_field_angle_count"], endpoint=False)
    frequencies = np.asarray(config["frequencies_hz"])
    result = None
    field = None
    attempts = []
    accepted = False
    for duration in config["duration_schedule_s"]:
        try:
            result = simulate_cuda(
                grid,
                objects,
                source,
                frequencies=frequencies,
                duration=duration,
                pml_thickness=config["pml_thickness_m"],
                monitor_bounds=monitor,
                samples=config["material_samples_per_axis"],
                pec_mode=config.get("pec_mode", "conformal"),
                device=device,
                dtype="float64",
            )
        except ValueError as error:
            if "Simulation requires" in str(error) and "exceeding" in str(error) and result is not None:
                attempts.append({"duration_s": duration, "status": "step_hard_limit", "reason": str(error)})
                break
            raise
        field = result.monitor.normalized_far_field(angles)
        tail = result.diagnostics["tail_peak_over_global_peak"]
        attempts.append(
            {
                "duration_s": duration,
                "tail_peak_over_global_peak": tail,
                "Nt": result.diagnostics["Nt"],
                "wall_seconds": result.diagnostics["wall_seconds"],
            }
        )
        if np.isfinite(field).all() and tail < config["tail_limit"]:
            accepted = True
            break
    assert result is not None and field is not None
    loss = {}
    reference = None
    if reference_path is not None:
        with np.load(reference_path) as arrays:
            reference = arrays["complex_far_field"].copy()
        loss = scattering_loss(field, reference)
    metrics = sparse_cluster_metrics(definition["scene"], domain=config["domain_m"])
    record = {
        "schema_version": 1,
        "case_id": definition["case_id"],
        "fingerprint": fingerprint,
        "definition": definition,
        "status": "accepted" if accepted else "hard_limit_unsettled",
        "accepted": accepted,
        "attempts": attempts,
        "sparse_metrics": metrics,
        **loss,
        **result.diagnostics,
    }
    _atomic_npz(
        arrays_path,
        x=grid.x,
        y=grid.y,
        frequencies=frequencies,
        angles=angles,
        complex_far_field=field,
        **({"reference_complex_far_field": reference} if reference is not None else {}),
    )
    _atomic_json(record_path, record)
    return record, False


def _load_records(config, output, phase, sources):
    records = []
    for definition in definitions(config, phase):
        reference_identity = None
        if phase == "candidates":
            reference_identity, _ = _reference_fingerprint(output, config, definition["scene"])
        fingerprint = _sha256_json(
            {
                "definition": definition,
                "protocol": {
                    key: config[key]
                    for key in (
                        "domain_m",
                        "frequencies_hz",
                        "far_field_angle_count",
                        "duration_schedule_s",
                        "pml_thickness_m",
                        "tail_limit",
                        "material_samples_per_axis",
                    )
                },
                "sources": sources,
                "reference": reference_identity,
            }
        )
        record_path, arrays_path = _case_paths(output, definition)
        record = _valid_cache(
            record_path, arrays_path, fingerprint, config["far_field_angle_count"]
        )
        if record is None:
            raise ValueError(f"Missing or stale {phase} case: {definition['case_id']}")
        records.append(record)
    return records


def summarize(config, output, sources, *, reference_sources=None, candidate_sources=None):
    reference_sources = sources if reference_sources is None else reference_sources
    candidate_sources = sources if candidate_sources is None else candidate_sources
    references = _load_records(config, output, "references", reference_sources)
    candidates = _load_records(config, output, "candidates", candidate_sources)
    ref_by_key = {
        (row["definition"]["scene"]["scene_id"], row["definition"]["cells"]): row
        for row in references
    }
    reference_reports = {}
    reference_losses = []
    coarse_cells, fine_cells = config["reference_budgets"][-2:]
    for scene in config["scenes"]:
        scene_id = scene["scene_id"]
        coarse = ref_by_key[(scene_id, coarse_cells)]
        fine = ref_by_key[(scene_id, fine_cells)]
        _, coarse_path = _case_paths(
            output,
            next(
                row
                for row in definitions(config, "references")
                if row["scene"]["scene_id"] == scene_id and row["cells"] == coarse_cells
            ),
        )
        _, fine_path = _case_paths(
            output,
            next(
                row
                for row in definitions(config, "references")
                if row["scene"]["scene_id"] == scene_id and row["cells"] == fine_cells
            ),
        )
        with np.load(coarse_path) as arrays:
            coarse_field = arrays["complex_far_field"].copy()
        with np.load(fine_path) as arrays:
            fine_field = arrays["complex_far_field"].copy()
        convergence = scattering_loss(fine_field, coarse_field)
        reference_losses.append(convergence["joint_scattering_loss"])
        reference_reports[scene_id] = {
            "coarse_cells": coarse_cells,
            "fine_cells": fine_cells,
            "both_settled": coarse["accepted"] and fine["accepted"],
            **convergence,
        }

    by_key = defaultdict(list)
    for row in candidates:
        definition = row["definition"]
        by_key[(definition["scene"]["scene_id"], definition["cells"])].append(row)
    exponent = config["ranking"]["nt_cost_exponent"]
    groups = []
    for scene in config["scenes"]:
        scene_metrics = sparse_cluster_metrics(scene, domain=config["domain_m"])
        for cells in config["budgets"]:
            rows = by_key[(scene["scene_id"], cells)]
            uniform = next(row for row in rows if row["definition"]["policy"]["name"] == "uniform")
            accepted = [row for row in rows if row["accepted"]]
            for row in accepted:
                row["soft_nt_score"] = row["joint_scattering_loss"] * (
                    row["Nt"] / uniform["Nt"]
                ) ** exponent
            best = min(accepted, key=lambda row: row["soft_nt_score"]) if accepted else None
            improvement = (
                uniform["joint_scattering_loss"] / best["soft_nt_score"]
                if uniform["accepted"] and best is not None
                else None
            )
            groups.append(
                {
                    "scene_id": scene["scene_id"],
                    "gap_stratum": scene["gap_stratum"],
                    "shape_topology": scene_metrics["shape_topology"],
                    "material_topology": scene_metrics["material_topology"],
                    "cells": cells,
                    "uniform_accepted": uniform["accepted"],
                    "best_policy": (
                        best["definition"]["policy"]["name"] if best is not None else None
                    ),
                    "uniform_loss": uniform.get("joint_scattering_loss"),
                    "best_soft_nt_score": best.get("soft_nt_score") if best is not None else None,
                    "improvement_over_uniform": improvement,
                    "meaningful_win": bool(
                        improvement is not None
                        and improvement >= config["gate"]["minimum_meaningful_improvement"]
                    ),
                }
            )

    low = [row for row in groups if row["cells"] in config["budgets"][:2]]
    improvements = [row["improvement_over_uniform"] for row in low if row["improvement_over_uniform"]]
    def summarize_strata(field):
        result = {}
        for name in sorted({row[field] for row in low}):
            selected = [row for row in low if row[field] == name]
            result[name] = {
                "group_count": len(selected),
                "meaningful_win_count": sum(row["meaningful_win"] for row in selected),
                "meaningful_win_fraction": float(
                    np.mean([row["meaningful_win"] for row in selected])
                ),
                "median_improvement_over_uniform": float(
                    np.median([row["improvement_over_uniform"] for row in selected])
                ),
            }
        return result

    gap_strata = summarize_strata("gap_stratum")
    shape_strata = summarize_strata("shape_topology")
    material_strata = summarize_strata("material_topology")
    checks = {
        "all_references_settled": all(row["accepted"] for row in references),
        "references_converged": max(reference_losses) <= config["gate"][
            "maximum_reference_convergence_loss"
        ],
        "all_uniform_baselines_settled": all(row["uniform_accepted"] for row in groups),
        "low_budget_meaningful_fraction": float(np.mean([row["meaningful_win"] for row in low]))
        >= config["gate"]["minimum_low_budget_meaningful_fraction"],
        "low_budget_median_improvement": float(np.median(improvements))
        >= config["gate"]["minimum_low_budget_median_improvement"],
        "each_gap_stratum_meaningful_fraction": all(
            row["meaningful_win_fraction"]
            >= config["gate"]["minimum_each_gap_stratum_meaningful_fraction"]
            for row in gap_strata.values()
        ),
        "each_shape_topology_meaningful_fraction": all(
            row["meaningful_win_fraction"]
            >= config["gate"].get("minimum_each_shape_topology_meaningful_fraction", 0)
            for row in shape_strata.values()
        ),
        "each_material_topology_meaningful_fraction": all(
            row["meaningful_win_fraction"]
            >= config["gate"].get("minimum_each_material_topology_meaningful_fraction", 0)
            for row in material_strata.values()
        ),
        "exact_axis_budgets": all(
            row["Nx"] == row["definition"]["cells"]
            and row["Ny"] == row["definition"]["cells"]
            for row in candidates
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "schema_version": 1,
        "decision": "passes_sparse_pair_headroom" if all(checks.values()) else "fails_sparse_pair_headroom",
        "checks": checks,
        "gate": config["gate"],
        "reference_case_count": len(references),
        "candidate_case_count": len(candidates),
        "reference_convergence": reference_reports,
        "low_budget": {
            "group_count": len(low),
            "meaningful_win_count": sum(row["meaningful_win"] for row in low),
            "meaningful_win_fraction": float(np.mean([row["meaningful_win"] for row in low])),
            "median_improvement_over_uniform": float(np.median(improvements)),
        },
        "by_gap_stratum": gap_strata,
        "by_shape_topology": shape_strata,
        "by_material_topology": material_strata,
        "groups": groups,
        "source_hashes": {
            "reference_phase": reference_sources,
            "candidate_phase": candidate_sources,
        },
    }
    _atomic_json(output / "report.json", report)
    return report


def main():
    args = parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    config_path = args.config.resolve()
    config = load_config(config_path)
    sources = source_hashes(config_path)
    if args.phase == "summarize":
        reference_sources = (
            json.loads(args.reference_sources.read_text()) if args.reference_sources else None
        )
        candidate_sources = (
            json.loads(args.candidate_sources.read_text()) if args.candidate_sources else None
        )
        print(
            json.dumps(
                summarize(
                    config,
                    args.output,
                    sources,
                    reference_sources=reference_sources,
                    candidate_sources=candidate_sources,
                ),
                indent=2,
            )
        )
        return
    cases = definitions(config, args.phase)
    selected = [row for index, row in enumerate(cases) if index % args.shards == args.shard]
    for index, definition in enumerate(selected, 1):
        record, cached = run_case(config, definition, args.output, args.device, sources)
        print(
            f"[{index}/{len(selected)}] {definition['case_id']} {record['status']} "
            f"Nt={record['Nt']} wall={record['wall_seconds']:.2f}s "
            f"{'cached' if cached else 'ran'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
