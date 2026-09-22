"""Record and apply the pre-epoch v6 sampling correction to a stopped workflow."""

import fcntl
import json
import shutil
import time
from pathlib import Path

from fdtdmesh.data.schema import provenance
from fdtdmesh.evaluation.references import _write_json


def main():
    root = Path("artifacts/training_v6_2000").resolve()
    archive = root / "sampling_migration"
    if archive.exists():
        raise ValueError("Sampling migration already exists")
    with (root / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive.mkdir()
        for relative in (
            "workflow.json",
            "provenance.json",
            "launch.json",
            "stage.json",
            "source_snapshot.tar.gz",
            "teacher/targets_run.json",
        ):
            source = root / relative
            if source.exists():
                destination = archive / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, destination)
        workflow = json.loads((archive / "workflow.json").read_text())
        if (root / "pretraining/training.json").exists():
            raise ValueError("Sampling migration is only valid before the first epoch")
        workflow["config"]["balance_budgets"] = False
        prov = provenance()
        workflow["source_sha256"] = prov["source_sha256"]
        workflow["native_binaries"] = prov["native_binaries"]
        _write_json(root / "workflow.json", workflow)
        _write_json(root / "provenance.json", prov)
        target_identity = json.loads((archive / "teacher/targets_run.json").read_text())
        target_identity["source_sha256"] = prov["source_sha256"]
        _write_json(root / "teacher/targets_run.json", target_identity)
        completed = json.loads((root / "teacher/progress.json").read_text())["completed"]
        _write_json(
            archive / "migration.json",
            dict(
                reason="avoid extreme repetition of the few feasible 32-grid targets",
                old_balance_budgets=True,
                new_balance_budgets=False,
                completed_target_records_preserved=completed,
                cnn_epochs_before_migration=0,
                source_sha256=prov["source_sha256"],
                migrated_unix=time.time(),
            ),
        )


if __name__ == "__main__":
    main()
