"""Run one scene with the current mesh CNN and save mesh/RCS figures.

All FDTD calls use the project's compiled CUDA backend. Torch is used only for
CNN inference; the FDTD time loop never uses Torch.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scattermesh.curriculum_v2 import DOMAIN, rasterize_v2, scene_metrics
from scattermesh.evaluation_v2 import load_checkpoint, predict_axes
from scattermesh.scoring_v2 import relative_l2
from scattermesh.simulation_v2 import FREQUENCIES, evaluate_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_RUN = PROJECT_ROOT / "runs_v2" / "c0_restart" / "acquisition_002"
SUPPORTED_BUDGETS = (32, 48, 64, 96, 128, 192, 256, 384, 512)
REFERENCE_LEVELS = (96, 128, 192, 256, 384, 512, 768, 1024)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _checkpoint_path(requested):
    if requested is not None:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CNN checkpoint does not exist: {path}")
        return path
    training = CURRENT_RUN / "large_cnn"
    frozen = training / "frozen_selection.json"
    if frozen.is_file():
        selected = Path(json.loads(frozen.read_text())["checkpoint"])
        if selected.is_file():
            return selected.resolve()
        raise FileNotFoundError(f"Frozen CNN checkpoint is missing: {selected}")
    paths = sorted(training.glob("checkpoint_epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(
            "The current C0 CNN has no checkpoint yet. Wait for acquisition and training, "
            "or pass --checkpoint /absolute/path/to/checkpoint.pt."
        )
    # The trainer retains its three best profile-validation checkpoints.
    def validation_loss(path):
        value = torch.load(path, map_location="cpu", weights_only=False)
        return float(value["validation_profile_loss"])

    return min(paths, key=validation_loss).resolve()


def _arrays(directory):
    with np.load(directory / "spectra.npz") as data:
        return {key: data[key].copy() for key in ("x", "y", "angles", "frequencies", "complex_far_field")}


def _run_case(scene, cells, angle, output, *, device, policy, axes=None, reference=None):
    record, _ = evaluate_case(
        scene,
        cells,
        angle,
        policy,
        output,
        device=device,
        selected_axes=axes,
        reference=reference,
    )
    spectra = output / "spectra.npz"
    if not spectra.is_file() or record.get("spectra_sha256") != hashlib.sha256(spectra.read_bytes()).hexdigest():
        raise RuntimeError(
            f"{output.name} simulation failed ({record['status']}): "
            f"{record.get('error', record.get('reason', 'no saved spectra'))}. "
            f"See {output / 'record.json'}"
        )
    return record, _arrays(output)


def _uniform_case(scene, cells, angle, output, *, device, reference=None):
    if cells <= 512:
        return _run_case(scene, cells, angle, output, device=device, policy="uniform", reference=reference)
    axis = np.linspace(0.0, DOMAIN, cells + 1)
    # The legacy candidate enumeration ends at 512. The learned-axis entry
    # accepts explicit uniform axes at larger reference resolutions.
    return _run_case(
        scene, cells, angle, output, device=device, policy="learned",
        axes=(axis, axis), reference=reference,
    )


def _refined_reference(scene, budget, angle, output, *, device, max_cells, complex_tol, width_tol):
    previous = None
    observations = []
    for cells in REFERENCE_LEVELS:
        if cells <= budget or cells > max_cells:
            continue
        directory = output / "reference" / f"n{cells}"
        record, arrays = _uniform_case(scene, cells, angle, directory, device=device)
        row = {
            "cells": cells,
            "status": record["status"],
            "accepted": bool(record.get("accepted")),
            "dt": record.get("dt"),
            "wall_seconds": record.get("wall_seconds"),
            "tail_ratio": record.get("tail_peak_over_global_peak"),
        }
        if not record.get("accepted"):
            observations.append(row)
            break
        if previous is not None:
            current_field = arrays["complex_far_field"]
            prior_field = previous["complex_far_field"]
            row["complex_change"] = relative_l2(current_field, prior_field)
            row["width_change"] = relative_l2(
                2 * np.pi * abs(current_field) ** 2,
                2 * np.pi * abs(prior_field) ** 2,
            )
            observations.append(row)
            if row["complex_change"] <= complex_tol and row["width_change"] <= width_tol:
                report = {
                    "status": "spatially_converged",
                    "selected_cells": cells,
                    "complex_tolerance": complex_tol,
                    "width_tolerance": width_tol,
                    "observations": observations,
                    "reference_uncertainty_indicator": max(row["complex_change"], row["width_change"]),
                    "qualification_scope": "successive uniform spatial refinement and field-tail settling",
                }
                _json(output / "reference_report.json", report)
                return report, arrays
        else:
            observations.append(row)
        previous = arrays
    report = {
        "status": "unsettled_or_not_spatially_converged",
        "selected_cells": None,
        "complex_tolerance": complex_tol,
        "width_tolerance": width_tol,
        "observations": observations,
    }
    _json(output / "reference_report.json", report)
    raise RuntimeError(
        "The automatic uniform reference did not settle and converge within "
        f"{max_cells} cells per axis. See {output / 'reference_report.json'}. "
        "Increase --max-reference-cells only if the extra cost is acceptable."
    )


def _plot_mesh(scene, arrays_by_name, output, title):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    raster = rasterize_v2(scene, resolution=512)
    material = np.zeros((512, 512), dtype=np.uint8)
    material[raster[0] > 0.5] = 1
    material[raster[3] > 0.5] = 2
    names = list(arrays_by_name)
    figure, axes = plt.subplots(1, len(names), figsize=(5.1 * len(names), 5.1), squeeze=False)
    for ax, name in zip(axes[0], names):
        values = arrays_by_name[name]
        ax.imshow(
            material, extent=(0, DOMAIN, 0, DOMAIN), origin="lower",
            cmap=ListedColormap(("#ffffff", "#96c4e6", "#5f6570")),
            vmin=0, vmax=2, interpolation="nearest",
        )
        count = len(values["x"]) - 1
        opacity = 0.35 if count <= 128 else 0.16 if count <= 384 else 0.08
        ax.vlines(values["x"], 0, DOMAIN, colors="#172b4d", linewidth=0.35, alpha=opacity)
        ax.hlines(values["y"], 0, DOMAIN, colors="#172b4d", linewidth=0.35, alpha=opacity)
        ax.set(xlim=(0, DOMAIN), ylim=(0, DOMAIN), aspect="equal", xlabel="x (m)", ylabel="y (m)")
        ax.set_title(f"{name} · {count} × {count} cells")
    figure.suptitle(title)
    figure.tight_layout()
    path = output / ("mesh_comparison.png" if len(names) > 1 else "mesh.png")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_rcs(arrays_by_name, output, title, frequency_index):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"projection": "polar"})
    colors = {"CNN": "#1261a0", "uniform": "#747980", "refined reference": "#c44e52"}
    for name, values in arrays_by_name.items():
        angles = values["angles"]
        width = 2 * np.pi * abs(values["complex_far_field"][frequency_index]) ** 2
        ax.plot(
            np.r_[angles, angles[0] + 2 * np.pi], np.r_[width, width[0]],
            label=name, color=colors.get(name), linewidth=2.0 if name == "CNN" else 1.6,
        )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(f"{title}\n2D scattering width at {FREQUENCIES[frequency_index] / 1e9:.1f} GHz", pad=22)
    ax.set_rlabel_position(135)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    figure.tight_layout()
    path = output / ("rcs_comparison_polar.png" if len(arrays_by_name) > 1 else "rcs_polar.png")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def run_example(scene, *, title=None, argv=None):
    """Load a trained CNN, run compiled CUDA FDTD, and save example figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=SUPPORTED_BUDGETS, default=64,
                        help="cells per x/y axis for CNN and uniform meshes (default: 64)")
    parser.add_argument("--compare", action="store_true",
                        help="first converge a finer uniform reference, then run CNN and uniform at the requested budget")
    parser.add_argument("--checkpoint", type=Path,
                        help="trained CNN checkpoint; default: current frozen or best C0 checkpoint")
    parser.add_argument("--device", default="cuda:0", help="GPU used for CNN inference and compiled CUDA FDTD")
    parser.add_argument("--angle-deg", type=float, default=0.0, help="plane-wave incidence angle in degrees")
    parser.add_argument("--frequency-ghz", type=float, choices=(0.8, 1.0, 1.2), default=1.0)
    parser.add_argument("--output", type=Path, help="result directory (default: runs_v2/examples/<scene>_n<budget>)")
    parser.add_argument("--max-reference-cells", type=int, choices=REFERENCE_LEVELS, default=1024)
    parser.add_argument("--reference-complex-tol", type=float, default=0.01)
    parser.add_argument("--reference-width-tol", type=float, default=0.02)
    args = parser.parse_args(argv)
    if not args.device.startswith("cuda:"):
        parser.error("All example FDTD runs require the compiled CUDA backend (--device cuda:N)")
    if not np.isfinite(args.angle_deg) or not 0 < args.reference_complex_tol < 1 or not 0 < args.reference_width_tol < 1:
        parser.error("Incidence angle and reference tolerances must be finite and valid")
    if args.compare and sum(args.budget < level <= args.max_reference_cells for level in REFERENCE_LEVELS) < 2:
        parser.error("Comparison requires at least two reference grids finer than the requested budget")
    scene_metrics(scene)
    checkpoint_path = _checkpoint_path(args.checkpoint)
    output = (args.output or PROJECT_ROOT / "runs_v2" / "examples" /
              f"{scene['lineage_id']}_n{args.budget}_{'comparison' if args.compare else 'cnn'}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "scene.json", scene)
    checkpoint, reference = str(checkpoint_path), None
    model, metadata = load_checkpoint(checkpoint_path, args.device)
    angle = float(np.deg2rad(args.angle_deg))
    reference_report = None
    mesh_arrays = {}
    rcs_arrays = {}
    if args.compare:
        reference_report, reference_arrays = _refined_reference(
            scene, args.budget, angle, output, device=args.device,
            max_cells=args.max_reference_cells,
            complex_tol=args.reference_complex_tol,
            width_tol=args.reference_width_tol,
        )
        reference = reference_arrays["complex_far_field"]
        mesh_arrays["refined reference"] = reference_arrays
        rcs_arrays["refined reference"] = reference_arrays
    predicted_axes, repairs = predict_axes(model, scene, args.budget, angle, FREQUENCIES, args.device)
    cnn_record, cnn_arrays = _run_case(
        scene, args.budget, angle, output / "cnn", device=args.device,
        policy="learned", axes=predicted_axes, reference=reference,
    )
    mesh_arrays["CNN"] = cnn_arrays
    rcs_arrays["CNN"] = cnn_arrays
    uniform_record = None
    if args.compare:
        uniform_record, uniform_arrays = _uniform_case(
            scene, args.budget, angle, output / "uniform", device=args.device,
            reference=reference,
        )
        mesh_arrays["uniform"] = uniform_arrays
        rcs_arrays["uniform"] = uniform_arrays
    display_title = title or scene.get("family", scene["lineage_id"])
    mesh_path = _plot_mesh(scene, mesh_arrays, output, display_title)
    rcs_path = _plot_rcs(rcs_arrays, output, display_title, (0.8, 1.0, 1.2).index(args.frequency_ghz))
    summary = {
        "scene": scene,
        "checkpoint": checkpoint,
        "checkpoint_epoch": metadata.get("epoch"),
        "budget_cells_per_axis": args.budget,
        "incidence_angle_deg": args.angle_deg,
        "frequency_ghz_plotted": args.frequency_ghz,
        "compiled_cuda_device": args.device,
        "projection_repair_fractions": list(repairs),
        "reference": reference_report,
        "cnn": {key: cnn_record.get(key) for key in ("status", "accepted", "dt", "Nt", "wall_seconds", "complex_relative_l2", "width_relative_l2", "joint_scattering_loss")},
        "uniform": ({key: uniform_record.get(key) for key in ("status", "accepted", "dt", "Nt", "wall_seconds", "complex_relative_l2", "width_relative_l2", "joint_scattering_loss")} if uniform_record is not None else None),
        "mesh_figure": str(mesh_path),
        "rcs_figure": str(rcs_path),
        "rcs_quantity": "2D scattering width = 2*pi*|far_field|^2 (RCS analogue)",
    }
    _json(output / "summary.json", summary)
    print(json.dumps({"mesh": str(mesh_path), "polar_rcs": str(rcs_path), "summary": str(output / 'summary.json'), "cnn_status": cnn_record["status"], "uniform_status": uniform_record["status"] if uniform_record is not None else None}, indent=2))
    return summary
