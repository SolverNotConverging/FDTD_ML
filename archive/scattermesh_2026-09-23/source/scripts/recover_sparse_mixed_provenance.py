#!/usr/bin/env python3
"""Reconstruct source snapshots used by the two mixed-pair pilot phases."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "scripts/run_sparse_pair_headroom_pilot.py"
CONFIG_PATH = ROOT / "configs/sparse_mixed_pair_headroom_pilot.json"
OUTPUT = ROOT / "runs/sparse_mixed_pair_headroom_pilot"


def _sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def _git_source(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def main():
    spec = importlib.util.spec_from_file_location("sparse_mixed_pilot", PILOT_PATH)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    config = pilot.load_config(CONFIG_PATH)
    current = pilot.source_hashes(CONFIG_PATH)

    before_input_schema = (ROOT / "src/scattermesh/__init__.py").read_text()
    for line in (
        "    conditioning_features_v2,\n",
        "    rasterize_scene,\n",
        "    resample_axis_profiles,\n",
        '    "rasterize_scene",\n',
        '    "conditioning_features_v2",\n',
        '    "resample_axis_profiles",\n',
    ):
        if line not in before_input_schema:
            raise ValueError(f"Cannot reconstruct pre-input-schema package: {line.strip()}")
        before_input_schema = before_input_schema.replace(line, "")
    reference = current.copy()
    reference["src/scattermesh/__init__.py"] = _sha256(before_input_schema.encode())
    for module in ("distillation", "model"):
        path = f"src/scattermesh/{module}.py"
        reference[path] = _sha256(_git_source(path))

    before_raster_cache = (ROOT / "src/scattermesh/model.py").read_text()
    for text in ("from collections import OrderedDict\n", "        self._raster_cache = OrderedDict()\n"):
        if text not in before_raster_cache:
            raise ValueError("Cannot reconstruct model before raster cache")
        before_raster_cache = before_raster_cache.replace(text, "")
    start = before_raster_cache.index('        key = example["geometry_id"]\n')
    end = before_raster_cache.index("        return {\n", start)
    before_raster_cache = before_raster_cache[:start] + (
        '        if self.input_schema == "circle_v1":\n'
        "            raster = rasterize_circle(example, self.resolution)\n"
        "            conditioning = conditioning_features(example)\n"
        "        else:\n"
        "            raster = rasterize_scene(example, self.resolution)\n"
        "            conditioning = conditioning_features_v2(example)\n"
    ) + before_raster_cache[end:]
    before_raster_cache = before_raster_cache.replace('    reduction="mean",\n', "")
    old_loss = '''    return (0.5 * selected + 0.5 * weighted).mean(), {
'''
    new_loss = '''    per_example = 0.5 * selected + 0.5 * weighted
    if reduction not in {"mean", "none"}:
        raise ValueError("Reduction must be mean or none")
    return (per_example.mean() if reduction == "mean" else per_example), {
'''
    if new_loss not in before_raster_cache:
        raise ValueError("Cannot reconstruct model before loss reduction change")
    before_raster_cache = before_raster_cache.replace(new_loss, old_loss)
    candidate = current.copy()
    candidate_init = (ROOT / "src/scattermesh/__init__.py").read_text()
    ordered = "    rasterize_circle,\n    rasterize_scene,\n    resample_axis_profiles,\n"
    launched = "    rasterize_scene,\n    resample_axis_profiles,\n    rasterize_circle,\n"
    if ordered not in candidate_init:
        raise ValueError("Cannot reconstruct package import order at candidate launch")
    candidate["src/scattermesh/__init__.py"] = _sha256(
        candidate_init.replace(ordered, launched).encode()
    )
    candidate["src/scattermesh/model.py"] = _sha256(before_raster_cache.encode())

    protocol = {
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
    }
    verification = {}
    for phase, sources in (("references", reference), ("candidates", candidate)):
        count = 0
        for definition in pilot.definitions(config, phase):
            reference_identity = None
            if phase == "candidates":
                reference_identity, _ = pilot._reference_fingerprint(
                    OUTPUT, config, definition["scene"]
                )
            fingerprint = pilot._sha256_json(
                {
                    "definition": definition,
                    "protocol": protocol,
                    "sources": sources,
                    "reference": reference_identity,
                }
            )
            record_path, arrays_path = pilot._case_paths(OUTPUT, definition)
            if pilot._valid_cache(
                record_path, arrays_path, fingerprint, config["far_field_angle_count"]
            ) is None:
                raise ValueError(f"Source reconstruction failed: {definition['case_id']}")
            count += 1
        path = OUTPUT / f"{phase}_source_hashes.json"
        pilot._atomic_json(path, sources)
        verification[phase] = {"verified_cases": count, "source_hashes_file": str(path)}
    pilot._atomic_json(OUTPUT / "provenance_recovery.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
