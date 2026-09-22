"""Migrate a stopped campaign to the probe/PEC roundoff fix without losing references."""

import argparse
import fcntl
import hashlib
import json
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

from fdtdmesh.data.schema import SceneSpec, digest, provenance
from fdtdmesh.evaluation.references import _write_json


def repair(output):
    output = Path(output).resolve()
    with ExitStack() as stack:
        for name in ("workflow.lock", "campaign.lock"):
            lock = stack.enter_context((output / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((output / "campaign.json").read_text())
        prov = provenance()
        if config["native_binaries"] != prov["native_binaries"]:
            raise ValueError("Native solver changed; this repair only allows Python anchor changes")
        pending = []
        for attempt in sorted(output.glob("lane*/attempts/*")):
            lane = attempt.parent.parent.name
            if (output / lane / "decisions" / f"{attempt.name}.json").exists():
                continue
            refs = attempt / "references"
            status_path = refs / attempt.name / "reference.json"
            if not status_path.exists():
                raise ValueError(f"Unexpected incomplete attempt: {attempt}")
            status = json.loads(status_path.read_text())
            errors = [
                entry.get("error")
                for duration in status.get("duration_history", [])
                for entry in duration.get("levels", [])
                if entry.get("error")
            ]
            if (
                status["status"] != "failed"
                or not errors
                or any(
                    error != "Uniform mesh cannot retain these anchors; use mesh_from_density"
                    for error in errors
                )
            ):
                raise ValueError(f"Unexpected failure requires separate investigation: {attempt}")
            spec = SceneSpec.from_dict(
                json.loads((attempt / "manifest.json").read_text())["scenes"][0]
            )
            if status["scene_hash"] != spec.content_hash:
                raise ValueError("Failed reference scene identity differs")
            for n in config["reference"]["reference_levels"]:
                sim = spec.build([n, n], reference=True)
                mesh = sim.mesh_uniform()
                for probe in [*(source[0] for source in sim.sources), *sim.receivers]:
                    if probe.x not in mesh.x or probe.y not in mesh.y:
                        raise ValueError("Built probe is not exactly anchored")
            pending.append(attempt)
        if not pending:
            raise ValueError("No matching failures to repair")
        archive = output / f"anchor_roundoff_repair_{time.time_ns()}"
        archive.mkdir()
        for name in (
            "campaign.json",
            "provenance.json",
            "workflow_failure.json",
            "workflow_stage.json",
            "workflow_launch.json",
            "progress.json",
            "workers.json",
        ):
            path = output / name
            if path.exists():
                shutil.copy2(path, archive / name)
        # Audit the exact bytes of every committed decision/reference to document
        # that completed convergence results are retained under their original provenance.
        preserved = {}
        for path in sorted(output.glob("lane*/decisions/*.json")):
            row = json.loads(path.read_text())
            for file in [path, *([output / row["reference"]] if row.get("reference") else [])]:
                preserved[str(file.relative_to(output))] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
        _write_json(archive / "preserved_sha256.json", preserved)
        for attempt in pending:
            target = archive / attempt.relative_to(output)
            target.mkdir(parents=True)
            shutil.move(str(attempt / "references"), target / "references")
            shutil.copy2(attempt / "manifest.json", target / "manifest.json")
            if (attempt / "worker.log").exists():
                shutil.copy2(attempt / "worker.log", target / "worker.log")
        new = {**config, "source_sha256": prov["source_sha256"]}
        new.pop("campaign_id")
        new["campaign_id"] = digest(new)
        _write_json(output / "campaign.json", new)
        _write_json(output / "provenance.json", prov)
        for name, expected in preserved.items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Completed record changed during repair: {name}")
        report = dict(
            old_campaign_id=config["campaign_id"],
            new_campaign_id=new["campaign_id"],
            retried_scenes=[p.name for p in pending],
            preserved_record_files=len(preserved),
            numerical_policy="built probes coincide with existing anchors within four ULPs",
            archive=str(archive),
            time=time.time(),
        )
        _write_json(archive / "repair.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(repair(args.output), indent=2))


if __name__ == "__main__":
    main()
