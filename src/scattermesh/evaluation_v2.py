"""Frozen-checkpoint physical evaluation and accuracy-led scene summaries."""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .campaign_v2 import atomic_json, scoring_fingerprint
from .candidates_v2 import CANDIDATE_NAMES
from .curriculum_v2 import conditioning_v2, rasterize_v2, scene_material_kind
from .model_v2 import AxisDensityUNetV2
from .profiles_v2 import probability_axis
from .qualification_v2 import qualify_compact_reference, qualify_reference
from .scoring_v2 import penalty_sensitivity, threshold_cell_savings
from .simulation_v2 import evaluate_case, numerical_source_hashes


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AxisDensityUNetV2(base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict_axes(model, scene, cells, angle, frequencies, device):
    raster = torch.from_numpy(rasterize_v2(scene))[None].to(device)
    conditioned = torch.from_numpy(conditioning_v2(scene, cells, cells, angle, frequencies))[
        None
    ].to(device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")
        ),
    ):
        prediction = model(raster, conditioned)[0].float().cpu().numpy()
    axes, repairs = [], []
    for profile in prediction:
        axis, repair = probability_axis(profile, cells, max_ratio=3)
        axes.append(axis)
        repairs.append(repair)
    return tuple(axes), tuple(repairs)


def evaluate_checkpoint(
    manifest_path,
    campaign_output,
    checkpoint_path,
    output,
    *,
    split="validation",
    device="cuda:0",
    deadline=None,
):
    if split not in ("validation", "test"):
        raise ValueError("Only new validation or frozen test lineages may be evaluated")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("protocol") == "compact_c8_c9_poc_v1":
        if manifest.get("numerical_source_hashes") != numerical_source_hashes():
            raise ValueError("Numerical implementation changed after compact campaign freeze")
        if manifest.get("scoring_fingerprint") != scoring_fingerprint():
            raise ValueError("Scoring implementation changed after compact campaign freeze")
    output = Path(output)
    campaign_output = Path(campaign_output)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    rows = []
    requested = 0
    evaluation_conditions = manifest.get("evaluation_conditions", {}).get(split, {})
    for scene in manifest["scenes"]:
        if scene["split"] != split:
            continue
        conditions = evaluation_conditions.get(scene["lineage_id"])
        if conditions is None:
            conditions = [
                {"angle_index": angle_index, "cells": cells}
                for angle_index in range(len(manifest["incidence_angles"]))
                for cells in manifest["budgets"]
            ]
        for condition in conditions:
            angle_index, cells = condition["angle_index"], condition["cells"]
            angle = manifest["incidence_angles"][angle_index]
            requested += 1
            qualification_path = (
                campaign_output / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.json"
            )
            if deadline is not None and time.time() >= deadline:
                rows.append(
                    {
                        "scene_id": scene["lineage_id"],
                        "angle_index": angle_index,
                        "cells": cells,
                        "status": "incomplete",
                    }
                )
                continue
            qualification = (
                json.loads(qualification_path.read_text()) if qualification_path.exists() else {}
            )
            if manifest.get("protocol") == "compact_c8_c9_poc_v1":
                qualification = qualify_compact_reference(
                    scene,
                    angle_index,
                    angle,
                    campaign_output,
                    manifest["reference_policy"],
                    calibration_profile_sha256=manifest["reference_policy_source_profile_sha256"],
                    device=device,
                    deadline=deadline,
                )
            elif not qualification.get("accepted"):
                qualification = qualify_reference(
                    scene, angle_index, angle, campaign_output, device=device, deadline=deadline
                )
            if not qualification.get("accepted"):
                rows.append(
                    {
                        "scene_id": scene["lineage_id"],
                        "family": scene["family"],
                        "material": scene_material_kind(scene),
                        "stage": scene["stage"],
                        "angle_index": angle_index,
                        "cells": cells,
                        "status": qualification.get("status", "reference_not_qualified"),
                    }
                )
                continue
            with np.load(qualification_path.with_suffix(".npz")) as arrays:
                reference = arrays["complex_far_field"].copy()
            uniform, _ = evaluate_case(
                scene,
                cells,
                angle,
                "uniform",
                campaign_output
                / "cases"
                / f"{scene['lineage_id']}_a{angle_index}_n{cells}_uniform",
                device=device,
                reference=reference,
            )
            heuristic, _ = evaluate_case(
                scene,
                cells,
                angle,
                "interface",
                campaign_output
                / "cases"
                / f"{scene['lineage_id']}_a{angle_index}_n{cells}_interface",
                device=device,
                reference=reference,
            )
            if deadline is not None and time.time() >= deadline:
                rows.append(
                    {
                        "scene_id": scene["lineage_id"],
                        "angle_index": angle_index,
                        "cells": cells,
                        "status": "incomplete",
                    }
                )
                continue
            axes, repairs = predict_axes(
                model, scene, cells, angle, manifest["frequencies_hz"], device
            )
            learned, _ = evaluate_case(
                scene,
                cells,
                angle,
                "learned",
                output / "cases" / f"{scene['lineage_id']}_a{angle_index}_n{cells}",
                device=device,
                reference=reference,
                selected_axes=axes,
            )
            rows.append(
                {
                    "scene_id": scene["lineage_id"],
                    "family": scene["family"],
                    "material": scene_material_kind(scene),
                    "stage": scene["stage"],
                    "angle_index": angle_index,
                    "cells": cells,
                    "uniform": uniform,
                    "heuristic": heuristic,
                    "learned": learned,
                    "projection_repairs": repairs,
                    "reference_cells": qualification["selected_cells"],
                    "reference_uncertainty_relative": qualification.get(
                        "reference_uncertainty_relative"
                    ),
                    "status": "accepted"
                    if learned.get("accepted")
                    and uniform.get("accepted")
                    and heuristic.get("accepted")
                    else "invalid",
                }
            )
        atomic_json(
            output / "summary.json",
            {
                "schema_version": 2,
                "split": split,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "checkpoint_epoch": checkpoint["epoch"],
                "requested_conditions": requested,
                "rows": rows,
                "terminal": False,
            },
        )
    valid = [row for row in rows if row["status"] == "accepted"]
    mean_raw = (
        float(np.mean([row["learned"]["joint_scattering_loss"] for row in valid]))
        if valid
        else None
    )
    complete = requested == sum(row.get("status") in ("accepted", "invalid") for row in rows)
    summary = {
        "schema_version": 2,
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "requested_conditions": requested,
        "valid_conditions": len(valid),
        "mean_learned_raw_accuracy_loss": mean_raw,
        "terminal": complete,
        "rows": rows,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def select_checkpoint(evaluation_summaries, destination):
    summaries = [json.loads(Path(path).read_text()) for path in evaluation_summaries]
    if not summaries or any(summary["split"] != "validation" for summary in summaries):
        raise ValueError("Three validation FDTD summaries are required")
    if len(summaries) != 3 or any(
        not summary["terminal"] or summary["valid_conditions"] != summary["requested_conditions"]
        for summary in summaries
    ):
        raise ValueError("All three checkpoints need complete valid validation FDTD")
    chosen = min(summaries, key=lambda row: row["mean_learned_raw_accuracy_loss"])
    selection = {
        "schema_version": 2,
        "selection_metric": "raw_physical_accuracy",
        "checkpoint": chosen["checkpoint"],
        "validation_raw_accuracy": chosen["mean_learned_raw_accuracy_loss"],
        "comparison": [
            {
                "checkpoint": row["checkpoint"],
                "mean_raw_accuracy": row["mean_learned_raw_accuracy_loss"],
            }
            for row in summaries
        ],
    }
    destination = Path(destination)
    if destination.exists() and json.loads(destination.read_text()) != selection:
        raise ValueError("Frozen model selection already differs")
    atomic_json(destination, selection)
    return selection


def summarize_test(test_summary, campaign_output, destination):
    campaign_output = Path(campaign_output)
    summary = json.loads(Path(test_summary).read_text())
    if summary["split"] != "test":
        raise ValueError("Expected frozen new-lineage test evaluation")
    rows = summary["rows"]
    valid = [row for row in rows if row["status"] == "accepted"]
    grouped = defaultdict(list)
    for row in valid:
        uniform, heuristic, learned = row["uniform"], row["heuristic"], row["learned"]
        improvement = uniform["joint_scattering_loss"] / max(
            learned["joint_scattering_loss"], 1e-30
        )
        grouped[(row["family"], row["material"])].append(improvement)
        row["raw_accuracy_improvement_ratio"] = improvement
        row["heuristic_raw_accuracy_improvement_ratio"] = uniform["joint_scattering_loss"] / max(
            heuristic["joint_scattering_loss"], 1e-30
        )
        row["cnn_over_heuristic_accuracy_ratio"] = heuristic["joint_scattering_loss"] / max(
            learned["joint_scattering_loss"], 1e-30
        )
    material = {}
    for kind in sorted({row["material"] for row in valid}):
        subset = [row for row in valid if row["material"] == kind]
        ratios = [row["raw_accuracy_improvement_ratio"] for row in subset]
        material[kind] = {
            "count": len(subset),
            "fraction_improved_at_least_5_percent": float(np.mean(np.asarray(ratios) >= 1.05))
            if ratios
            else None,
            "median_improvement_ratio": float(np.median(ratios)) if ratios else None,
        }
    cohort_lineages = defaultdict(lambda: defaultdict(list))
    for row in valid:
        cohort_lineages[row["stage"]][row["scene_id"]].append(row["raw_accuracy_improvement_ratio"])
    by_target_cohort = {}
    for cohort in ("C8", "C9"):
        scene_ratios = [
            float(np.median(values)) for values in cohort_lineages[cohort].values() if values
        ]
        by_target_cohort[cohort] = {
            "lineages_with_valid_conditions": len(scene_ratios),
            "median_per_lineage_improvement_ratio": (
                float(np.median(scene_ratios)) if scene_ratios else None
            ),
            "fraction_of_lineages_improving_at_least_5_percent": (
                float(np.mean(np.asarray(scene_ratios) >= 1.05)) if scene_ratios else None
            ),
        }
    positive_demonstration = bool(
        summary["terminal"]
        and summary["valid_conditions"] == summary["requested_conditions"]
        and all(
            by_target_cohort[cohort]["lineages_with_valid_conditions"] > 0
            and by_target_cohort[cohort]["median_per_lineage_improvement_ratio"] > 1.0
            for cohort in ("C8", "C9")
        )
    )
    savings = {}
    for scene_id in set(row["scene_id"] for row in valid):
        for angle_index in (0, 1):
            group = [
                row
                for row in valid
                if row["scene_id"] == scene_id and row["angle_index"] == angle_index
            ]
            if group:
                savings[f"{scene_id}_a{angle_index}"] = threshold_cell_savings(
                    [row["uniform"] for row in group], [row["learned"] for row in group]
                )
    curves = defaultdict(list)
    performance = defaultdict(list)
    sensitivity = {}
    for row in valid:
        key = f"{row['family']}/{row['material']}/n{row['cells']}"
        uniform, heuristic, learned = row["uniform"], row["heuristic"], row["learned"]
        curves[key].append(
            {
                "uniform_complex_error": uniform["complex_relative_l2"],
                "heuristic_complex_error": heuristic["complex_relative_l2"],
                "learned_complex_error": learned["complex_relative_l2"],
                "uniform_width_error": uniform["width_relative_l2"],
                "heuristic_width_error": heuristic["width_relative_l2"],
                "learned_width_error": learned["width_relative_l2"],
                "improvement_ratio": row["raw_accuracy_improvement_ratio"],
                "reference_uncertainty_relative": row.get("reference_uncertainty_relative"),
            }
        )
        performance[key].append(
            {
                method: {
                    "dt": record.get("dt"),
                    "Nt": record.get("Nt"),
                    "wall_seconds": record.get("wall_seconds"),
                    "peak_cuda_memory_bytes": record.get("peak_cuda_memory_bytes"),
                }
                for method, record in (
                    ("uniform", uniform),
                    ("heuristic", heuristic),
                    ("learned", learned),
                )
            }
        )
        candidates = []
        for name in CANDIDATE_NAMES:
            path = (
                campaign_output
                / "cases"
                / f"{row['scene_id']}_a{row['angle_index']}_n{row['cells']}_{name}"
                / "record.json"
            )
            record = json.loads(path.read_text()) if path.exists() else {}
            candidates.append(
                {
                    "name": name,
                    "accepted": record.get("accepted", False),
                    "joint_scattering_loss": record.get("joint_scattering_loss"),
                    "dt": record.get("dt"),
                }
            )
        try:
            sensitivity[f"{row['scene_id']}_a{row['angle_index']}_n{row['cells']}"] = (
                penalty_sensitivity(candidates)
            )
        except ValueError:
            pass
    statuses = Counter()
    for path in campaign_output.glob("cases/*/record.json"):
        statuses[json.loads(path.read_text()).get("status", "unknown")] += 1
    reference_statuses = Counter()
    for path in campaign_output.glob("qualifications/*.json"):
        reference_statuses[json.loads(path.read_text()).get("status", "unknown")] += 1
    report = {
        "schema_version": 2,
        "test_complete": summary["terminal"]
        and summary["valid_conditions"] == summary["requested_conditions"],
        "requested_test_conditions": summary["requested_conditions"],
        "valid_test_conditions": len(valid),
        "by_material": material,
        "by_target_cohort": by_target_cohort,
        "by_shape_material": {
            f"{family}/{kind}": {
                "count": len(values),
                "median_improvement_ratio": float(np.median(values)),
            }
            for (family, kind), values in grouped.items()
        },
        "threshold_cell_savings": savings,
        "physical_error_curves": dict(curves),
        "time_step_runtime_memory": dict(performance),
        "teacher_penalty_sensitivity": sensitivity,
        "candidate_status_counts": dict(statuses),
        "reference_status_counts": dict(reference_statuses),
        "positive_demonstration": positive_demonstration,
        "positive_dielectric_gate": positive_demonstration,
        "rows": rows,
    }
    atomic_json(destination, report)
    return report


def compare_test_models(large_report, small_report, destination):
    """Compare frozen models only after both new-lineage test reports exist."""
    reports = [json.loads(Path(path).read_text()) for path in (large_report, small_report)]
    indexed = []
    for report in reports:
        indexed.append(
            {
                (row["scene_id"], row["angle_index"], row["cells"]): row
                for row in report["rows"]
                if row["status"] == "accepted"
            }
        )
    shared = sorted(set(indexed[0]) & set(indexed[1]))
    ratios = [
        indexed[1][key]["learned"]["joint_scattering_loss"]
        / max(indexed[0][key]["learned"]["joint_scattering_loss"], 1e-30)
        for key in shared
    ]
    comparison = {
        "schema_version": 2,
        "large_test_complete": reports[0]["test_complete"],
        "small_test_complete": reports[1]["test_complete"],
        "shared_valid_conditions": len(shared),
        "median_small_to_large_raw_error_ratio": float(np.median(ratios)) if ratios else None,
        "large_positive_demonstration": reports[0]["positive_demonstration"],
        "small_positive_demonstration": reports[1]["positive_demonstration"],
    }
    atomic_json(destination, comparison)
    return comparison


def render_test_figures(
    report_path, manifest_path, campaign_output, evaluation_output, destination
):
    """Save representative same-budget meshes and complex scattering curves."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    report = json.loads(Path(report_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    scenes = {scene["lineage_id"]: scene for scene in manifest["scenes"]}
    rows = report["rows"]
    valid = [row for row in rows if row["status"] == "accepted"]
    selected = []
    if valid:
        selected.append(("best", max(valid, key=lambda row: row["raw_accuracy_improvement_ratio"])))
        selected.append(
            ("worst", min(valid, key=lambda row: row["raw_accuracy_improvement_ratio"]))
        )
    invalid = next((row for row in rows if row.get("status") == "invalid" and "cells" in row), None)
    if invalid is not None:
        selected.append(("invalid", invalid))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    figures = []
    for label, row in selected:
        scene_id, angle_index, cells = row["scene_id"], row["angle_index"], row["cells"]
        key = f"{scene_id}_a{angle_index}_n{cells}"
        uniform_path = Path(campaign_output) / "cases" / f"{key}_uniform" / "spectra.npz"
        heuristic_path = Path(campaign_output) / "cases" / f"{key}_interface" / "spectra.npz"
        learned_path = Path(evaluation_output) / "cases" / key / "spectra.npz"
        reference_path = Path(campaign_output) / "qualifications" / f"{scene_id}_a{angle_index}.npz"
        if not all(
            path.exists() for path in (uniform_path, heuristic_path, learned_path, reference_path)
        ):
            figures.append({"label": label, "condition": key, "status": "spectra_unavailable"})
            continue
        with np.load(uniform_path) as data:
            x_u, y_u, field_u = (data[name].copy() for name in ("x", "y", "complex_far_field"))
        with np.load(heuristic_path) as data:
            x_h, y_h, field_h = (data[name].copy() for name in ("x", "y", "complex_far_field"))
        with np.load(learned_path) as data:
            x_l, y_l, field_l = (data[name].copy() for name in ("x", "y", "complex_far_field"))
        with np.load(reference_path) as data:
            field_ref = data["complex_far_field"].copy()
        occupation = rasterize_v2(scenes[scene_id], resolution=256)
        image = occupation[0] + occupation[3]
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for axis, x, y, title in (
            (axes[0, 0], x_u, y_u, "Uniform"),
            (axes[0, 1], x_h, y_h, "Fixed interface"),
            (axes[0, 2], x_l, y_l, "Learned"),
        ):
            axis.imshow(
                image, origin="lower", extent=(0, 1.2, 0, 1.2), cmap="Greys", vmin=0, vmax=1
            )
            axis.vlines(x, 0, 1.2, color="tab:blue", alpha=0.25, linewidth=0.4)
            axis.hlines(y, 0, 1.2, color="tab:orange", alpha=0.25, linewidth=0.4)
            axis.set(title=f"{title} {cells}×{cells}", xlabel="x (m)", ylabel="y (m)")
        angle = np.linspace(0, 360, field_ref.shape[-1], endpoint=False)
        for field, title, color in (
            (field_ref, "Reference", "black"),
            (field_u, "Uniform", "tab:blue"),
            (field_h, "Fixed interface", "tab:green"),
            (field_l, "Learned", "tab:orange"),
        ):
            axes[1, 0].plot(angle, 2 * np.pi * abs(field[1]) ** 2, label=title, color=color)
            axes[1, 1].plot(angle, field[1].real, label=title, color=color)
            axes[1, 2].plot(angle, field[1].imag, label=title, color=color)
        axes[1, 0].set(xlabel="Observation angle (degrees)", ylabel="2D scattering width (m)")
        axes[1, 1].set(xlabel="Observation angle (degrees)", ylabel="Real far field (sqrt(m))")
        axes[1, 2].set(xlabel="Observation angle (degrees)", ylabel="Imaginary far field (sqrt(m))")
        axes[1, 0].legend()
        figure.suptitle(f"{label}: {scene_id}, incidence {angle_index}, {row['status']}")
        target = destination / f"{label}_{key}.png"
        figure.savefig(target, dpi=180)
        plt.close(figure)
        figures.append({"label": label, "condition": key, "status": "saved", "path": str(target)})
    atomic_json(destination / "figures.json", figures)
    return figures
