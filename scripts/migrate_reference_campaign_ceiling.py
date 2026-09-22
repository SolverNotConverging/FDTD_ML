"""Lower a stopped v6 campaign's refinement ceiling without losing completed work."""

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

from fdtdmesh.data.campaign import summarize
from fdtdmesh.data.schema import digest, provenance
from fdtdmesh.evaluation.references import _write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-reference-level", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    ceiling = args.max_reference_level
    archive = output / f"policy_migration_max_{ceiling}"
    if archive.exists():
        raise ValueError(f"Migration archive already exists: {archive}")

    with (output / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / "campaign.json"
        old_config = json.loads(config_path.read_text())
        old_levels = old_config["reference"]["reference_levels"]
        new_levels = [level for level in old_levels if level <= ceiling]
        if new_levels == old_levels or new_levels[-1] != ceiling:
            raise ValueError("Requested ceiling does not lower the existing refinement sequence")

        archive.mkdir()
        for name in (
            "campaign.json",
            "provenance.json",
            "progress.json",
            "launch.json",
            "workers.json",
        ):
            source = output / name
            if source.exists():
                shutil.copy2(source, archive / name)

        reclassified = []
        for decision_path in sorted(output.glob("lane*/decisions/*.json")):
            decision = json.loads(decision_path.read_text())
            if decision.get("status") != "converged":
                continue
            reference_path = output / decision["reference"]
            reference = json.loads(reference_path.read_text())
            accepted = reference.get("accepted_budget")
            if accepted is None or max(accepted) <= ceiling:
                continue
            decision.update(
                status="nonconverged",
                terminal_reason=f"accepted_above_{ceiling}_policy_ceiling",
                previous_status="converged",
                previous_accepted_budget=accepted,
                policy_migration=f"max_reference_level_{ceiling}",
            )
            _write_json(decision_path, decision)
            reclassified.append(decision["scene_id"])

        interrupted = []
        interrupted_root = archive / "interrupted_attempts"
        for attempt in sorted(output.glob("lane*/attempts/*")):
            lane = attempt.parent.parent.name
            decision_path = output / lane / "decisions" / f"{attempt.name}.json"
            references = attempt / "references"
            if decision_path.exists() or not references.exists():
                continue
            destination = interrupted_root / lane / attempt.name / "references"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(references), destination)
            for name in ("config.json", "worker.log", "process.json"):
                source = attempt / name
                if source.exists():
                    shutil.copy2(source, destination.parent / name)
            interrupted.append(f"{lane}/{attempt.name}")

        prov = provenance()
        new_config = {
            **old_config,
            "reference": {
                **old_config["reference"],
                "reference_levels": new_levels,
                "extend_nonconverged": False,
            },
            "source_sha256": prov["source_sha256"],
            "native_binaries": prov["native_binaries"],
        }
        new_config.pop("campaign_id", None)
        new_config["campaign_id"] = digest(new_config)
        _write_json(config_path, new_config)
        _write_json(output / "provenance.json", prov)

        summary = summarize(output)
        _write_json(
            output / "progress.json",
            {
                "campaign_id": new_config["campaign_id"],
                "targets": new_config["targets"],
                "updated_unix": time.time(),
                **{key: value for key, value in summary.items() if key != "decisions"},
            },
        )
        report = {
            "old_campaign_id": old_config["campaign_id"],
            "new_campaign_id": new_config["campaign_id"],
            "old_reference_levels": old_levels,
            "new_reference_levels": new_levels,
            "extend_nonconverged": False,
            "reclassified_scene_ids": reclassified,
            "interrupted_attempts_archived": interrupted,
            "accepted_after_migration": summary["accepted"],
            "migrated_unix": time.time(),
        }
        _write_json(archive / "migration.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
