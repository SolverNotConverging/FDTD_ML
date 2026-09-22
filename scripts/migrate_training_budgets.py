"""Replace the pre-epoch 32-grid target stratum with 128, preserving cached work."""

import fcntl
import json
import shutil
import time
from pathlib import Path

from fdtdmesh.data.schema import provenance
from fdtdmesh.evaluation.references import _write_json


def main():
    root = Path("artifacts/training_v6_2000").resolve()
    archive = root / "budget_migration_32_to_128"
    if archive.exists():
        raise ValueError("Budget migration already exists")
    with (root / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "pretraining/training.json").exists():
            raise ValueError("Budget migration is only valid before the first CNN epoch")
        archive.mkdir()
        for relative in (
            "workflow.json",
            "provenance.json",
            "launch.json",
            "stage.json",
            "source_snapshot.tar.gz",
            "teacher/targets_run.json",
            "teacher/progress.json",
        ):
            source = root / relative
            if source.exists():
                destination = archive / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, destination)
        prov = provenance()
        workflow = json.loads((archive / "workflow.json").read_text())
        if workflow["budgets"] != [32, 48, 64, 96]:
            raise ValueError("Unexpected original workflow budgets")
        workflow["budgets"] = [48, 64, 96, 128]
        workflow["source_sha256"] = prov["source_sha256"]
        workflow["native_binaries"] = prov["native_binaries"]
        _write_json(root / "workflow.json", workflow)
        _write_json(root / "provenance.json", prov)
        targets = json.loads((archive / "teacher/targets_run.json").read_text())
        if targets["budgets"] != [[32, 32], [48, 48], [64, 64], [96, 96]]:
            raise ValueError("Unexpected original target budgets")
        targets["budgets"] = [[48, 48], [64, 64], [96, 96], [128, 128]]
        targets["source_sha256"] = prov["source_sha256"]
        _write_json(root / "teacher/targets_run.json", targets)
        old_progress = json.loads((archive / "teacher/progress.json").read_text())
        _write_json(
            archive / "migration.json",
            dict(
                old_budgets=[32, 48, 64, 96],
                new_budgets=[48, 64, 96, 128],
                completed_records_before_migration=old_progress["completed"],
                cached_32_records_preserved_but_excluded=True,
                cached_48_64_96_records_reused=True,
                cnn_epochs_before_migration=0,
                source_sha256=prov["source_sha256"],
                migrated_unix=time.time(),
            ),
        )


if __name__ == "__main__":
    main()
