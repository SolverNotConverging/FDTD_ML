# Restartable multi-GPU reference scheduler

`scripts/run_multi_gpu_campaign.py` executes an immutable JSON plan with one active
attempt per assigned GPU. Device assignment is stable because the device is part
of each attempt fingerprint. The coordinator never silently moves a task to a
different GPU after restart.

Each task invokes the atomic single-attempt dielectric runner. Before launch and
after every child exit, the coordinator checks the exact scene controls, device,
current `scattermesh` source hashes, configuration fingerprint, and finite native
complex spectra. A stale or corrupted result is invalidated and regenerated rather
than accepted because its dimensions or state entry happen to match.

The campaign directory contains:

- `campaign.lock`, preventing duplicate coordinators;
- `state.json`, atomically updated after every launch and completion;
- `summary.json`, the terminal outcome counts and task records;
- `logs/<task_id>.log`, preserving every subprocess attempt.

Scientific `pass` becomes scheduler status `succeeded`. Scientific `fail` becomes
terminal `nonconverged` and is retained without process retry. A missing artifact or
nonzero process exit is retried up to `--max-attempts`, then recorded as `failed`.
After interruption, any task marked `running` is reconciled against a valid result
first and otherwise returned to its pinned device queue.

Run the tracked example plan in the foreground:

```bash
.venv/bin/python scripts/run_multi_gpu_campaign.py \
  --plan configs/dielectric_reference_plan.example.json \
  --output runs/dielectric_reference_campaign/coordinator \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Read progress without taking the campaign lock:

```bash
.venv/bin/python scripts/run_multi_gpu_campaign.py \
  --plan configs/dielectric_reference_plan.example.json \
  --output runs/dielectric_reference_campaign/coordinator --status
```

The scheduler was exercised on all four physical GPUs with four concurrent 32²
attempts. All processes exited successfully in one attempt, produced independently
fingerprinted artifacts, and were recorded as scientifically nonconverged as
expected for the deliberately underresolved 12–13 ns smoke inputs. A second launch
performed no GPU work and retained every attempt count at one. Unit integration also
verifies deterministic round-robin assignment, explicit pinning, nonconvergence
retention, two-attempt process failure, immutable-plan rejection, corrupted-array
invalidation, and resume.

This stage schedules declared attempts; it does not invent convergence escalation
policy. Production scene generation must create a plan containing the required
spatial, duration, quadrature, and contour probes or a later policy layer must add
those attempts explicitly.
