"""Start the v6 small-CNN training workflow with resumable teacher preparation."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

from fdtdmesh.budget_schedule import mixed_budget_plan
from fdtdmesh.data.schema import provenance
from fdtdmesh.data.supplement import audit_campaign
from fdtdmesh.evaluation.references import _write_json
from fdtdmesh.teacher_campaign import prepare_targets
from fdtdmesh.training import TrainingConfig, train_model


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    campaign = args.campaign.resolve()
    manifest_path = campaign / "accepted_manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = TrainingConfig(
            width=16,
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            repair_weight=0,
            projection_samples=4,
            balance_budgets=False,
            sparse_sample_weight=args.sparse_sample_weight,
            early_stopping_patience=15,
            seed=2026,
        )
        prov = provenance()
        identity = dict(
            campaign=str(campaign),
            manifest=str(manifest_path),
            config=asdict(config),
            budgets=[48, 64, 96, 128],
            budget_policy=args.budget_policy,
            workers=args.workers,
            gpu=args.gpu,
            source_sha256=prov["source_sha256"],
            native_binaries=prov["native_binaries"],
        )
        identity_path = output / "workflow.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Training workflow identity changed; choose a new output")
        _write_json(identity_path, identity)
        _write_json(output / "provenance.json", prov)
        if not (output / "source_snapshot.tar.gz").exists():
            with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
                for folder in (root / "src", root / "scripts", root / "tests"):
                    for path in sorted(folder.rglob("*")):
                        if path.is_file() and path.suffix in (".py", ".pyx", ".cu", ".h"):
                            archive.add(path, arcname=str(path.relative_to(root)))
                for name in ("pyproject.toml", "uv.lock"):
                    archive.add(root / name, arcname=name)
        _write_json(output / "stage.json", dict(stage="reference_audit", updated_unix=time.time()))
        _, scenes, audit = audit_campaign(campaign)
        _write_json(
            output / "reference_audit.json",
            dict(
                **audit,
                verified_unix=time.time(),
            ),
        )
        print(f"Verified {audit['counts']} accepted references", flush=True)
        plan = (
            mixed_budget_plan(scenes, seed=config.seed) if args.budget_policy == "mixed" else None
        )
        if plan is not None:
            _write_json(output / "budget_plan.json", plan)
            print(
                f"Prepared {sum(map(len, plan['assignments'].values()))} reproducible budget assignments",
                flush=True,
            )
        _write_json(output / "stage.json", dict(stage="teacher_targets", updated_unix=time.time()))
        targets = prepare_targets(
            manifest_path,
            output / "teacher",
            workers=args.workers,
            budgets=tuple(identity["budgets"]),
            budget_plan=plan,
        )
        _write_json(output / "stage.json", dict(stage="cnn_pretraining", updated_unix=time.time()))
        print("Starting width-16 CNN from scratch (or resuming its saved optimizer)", flush=True)
        training = output / "pretraining"
        resume = training / "resume.pt"
        report = train_model(
            manifest_path,
            targets,
            training,
            config=config,
            device="cuda:0",
            resume=resume if resume.exists() else None,
            allow_dirty=True,
        )
        _write_json(
            output / "stage.json",
            dict(
                stage="pretraining_complete",
                updated_unix=time.time(),
                epochs=len(report["history"]),
                best_validation_loss=report["best_validation_loss"],
                next_stage="reference-based candidate search and physics distillation",
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("artifacts/reference_v6_2000"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/training_v6_2000"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sparse-sample-weight", type=float, default=2.0)
    parser.add_argument("--budget-policy", choices=("fixed", "mixed"), default="fixed")
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    if args.foreground:
        try:
            run(args)
        except Exception as error:
            _write_json(
                args.output / "failure.json",
                dict(
                    error=str(error),
                    error_type=type(error).__name__,
                    time=time.time(),
                ),
            )
            raise
        return
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:" + env.get("LD_LIBRARY_PATH", "")
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--foreground",
        "--campaign",
        str(args.campaign.resolve()),
        "--output",
        str(args.output.resolve()),
        "--workers",
        str(args.workers),
        "--gpu",
        str(args.gpu),
        "--sparse-sample-weight",
        str(args.sparse_sample_weight),
        "--budget-policy",
        args.budget_policy,
    ]
    with (args.output / "workflow.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _write_json(
        args.output / "launch.json", dict(pid=process.pid, command=command, time=time.time())
    )
    print(f"Training workflow PID {process.pid}; output {args.output.resolve()}")


if __name__ == "__main__":
    main()
