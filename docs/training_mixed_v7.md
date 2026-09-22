# Mixed-budget CNN training on 3,000 accepted reference scenes

## Physics pilot after pretraining

Launch or resume with `.venv/bin/python scripts/start_physics_pilot_v7.py`;
read progress with `python3 scripts/start_physics_pilot_v7.py --status`.
The output is `artifacts/physics_pilot_mixed_v7`. This evaluates the completed
mixed-budget `pretraining/best.pt` checkpoint without changing its weights.

Select the first validation scene by ID in each of the eight dense/sparse
families, independently of teacher feasibility or prediction error. Compare
true uniform, quasi-uniform, heuristic and CNN meshes at 64×64, 128×128,
80×80, 112×112, 72×104 and 104×72: 192 candidate runs across four GPU workers.
The last four pairs were excluded from gradient updates but participated in
validation/model selection; this pilot is not an untouched final test.

Reuse each scene's audited converged reference, actual extended duration,
observation times, frequency grid and averaging configuration. Candidate
cell updates are capped at 256 billion; meshing keeps the reference policy's
30-second projection limit. True uniform snaps PEC to mesh lines; the other
three retain PEC, source and receiver anchors. Record waveform and spectrum
errors, tail status, solver diagnostics and total wall time, plus mesh lines
and candidate waveforms for later plots. Projection/resource failures remain
explicit outcomes, not convergence claims or silently excluded scenes.

`plan.json` binds the checkpoint, code, selected scenes and reference hashes.
Each candidate saves an atomic JSON result; resuming skips every recorded
outcome, including failures. `report.json` aggregates progress and worker exit
codes every five seconds. No search or distillation starts automatically.

The completed combined corpus contains 2,400 training scenes, 300 validation
scenes and 300 IID test scenes. Dataset ID:
`0299d1f1c92765707a637a44b6906147912a282b4a9ef901f254684169386ee2`.
All 1,000 sparse references converged (989 at 1024² and 11 at 2048²).

## Budget schedule

Every training scene receives eight reproducibly chosen budget pairs:

- Fixed squares: 48², 64², 96² and 128².
- Two random intermediate squares: one count from 49–87 and one from 88–127,
  excluding fixed and reserved square counts.
- One random rectangle and its transpose, with both counts in [48, 128].

The seed stream combines seed 2026 and the stable scene seed. Ordering and scene
count do not affect a scene's schedule. Exact pairs **80², 112², 72×104 and 104×72**
are globally excluded from training. Each validation scene receives the eight
ordinary pairs and those four reserved pairs. They are held out from gradient
updates but participate in validation/model selection, not an untouched final
test. IID scenes remain excluded from both target preparation and training.

There are **19,200 requested train targets and 3,600 validation targets**. A pair
that fails mesh projection is recorded with its error; failure does not assert
mathematical infeasibility when optimization timed out. Coverage by split, family
and budget group is saved in `teacher/coverage.json` after preparation. Cache
filenames contain both Nx and Ny. The serialized plan and its digest bind resume
and target metadata, preventing reuse under a changed assignment.

## Workflow

```bash
.venv/bin/python scripts/start_training_v6.py \
  --campaign artifacts/reference_combined_v7_3000 \
  --output artifacts/training_mixed_v7_3000 \
  --budget-policy mixed --sparse-sample-weight 2 --workers 12 --gpu 0
```

First audit accepted reference identities and finite waveforms/spectra, then prepare
teacher targets with 12 CPU workers and a 30-second projection time limit per axis.
References are reused; no FDTD stepping is needed to generate these teacher meshes.
After preparation, initialize the width-16 CNN from scratch on GPU 0: batch 32,
AdamW at 1e-3, 100-epoch ceiling, patience 15, repair loss disabled during teacher
pretraining. PEC/source/receiver anchors and PML constraints remain enforced in
every teacher mesh. Sparse training pairs receive twice the sampling weight of
dense pairs; validation is unweighted. Check actual feasible coverage when
interpreting the resulting sampling mix.

Validation reports CDF target loss by budget group (including reserved budgets)
and by scene family. The saved best checkpoint uses the aggregate validation loss
over all feasible validation pairs. These are mesh-density errors, not waveform
errors. A later physical pilot must compare seen/unseen budgets and sparse/dense
scenes before launching more physics search/distillation. The earlier physics
distillation repair-step projection timeout remains a separate prerequisite for
that later stage.

```bash
python3 scripts/training_status.py
python3 scripts/training_status.py --watch 10
```

The detached workflow records its source snapshot, configuration, plan and logs in
`artifacts/training_mixed_v7_3000/`. It stops after CNN pretraining.

Validation before launch: 34 tests passed across budget scheduling, teacher cache,
training/resume, sparse weighting and physics regressions. The rectangular-cache
test generates real projected targets with shared Nx and different Ny, verifies
probability mass, checks resume identity, and exercises grouped validation.
