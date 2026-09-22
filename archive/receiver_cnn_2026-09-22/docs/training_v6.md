# Small-CNN training on the completed v6 corpus

For the forthcoming 3,000-scene dense+sparse corpus, see
[sparse supplement v7](sparse_supplement_v7.md). The launcher now audits split
counts against campaign metadata rather than requiring exactly 2,000 scenes.
Use a new output directory when changing the corpus. The physics-search launcher
also discovers the four additional sparse families.

The reference campaign completed with 2,000 accepted scenes: 1,600 training,
200 validation, and 200 IID test. Its dataset ID is
`5853cbb6e485306572c5e589eb843f32f8c937eee04031a79b2fb1612fc14497`.
The training workflow was launched on 2026-09-20 under
`artifacts/training_v6_2000/`.

## Initial training phase

1. Check completion identity, scene hashes, accepted resolution at most 2048,
   and finite waveform/spectrum arrays for all accepted references.
2. Generate legal heuristic mesh targets for train and validation only, with
   budgets 48/64/96/128, all PEC/probe anchors, PML collars and 1.4 grading.
   Twelve CPU workers process the 7,200 requested scene-budget pairs. Each pair
   is cached independently, including infeasibility and optimization timeouts;
   a timeout is not presented as proof that no legal mesh exists. Projection
   uses the existing 30-second optimization limit per axis.
3. Automatically initialize a width-16 ResU-Net from scratch and train on GPU 0.
   It has 128,578 parameters and ten raster input channels. AdamW uses batch 32,
   learning rate 1e-3 and weight decay 1e-5, with seed 2026. Samples are drawn
   uniformly from feasible scene-budget pairs. CDF target loss is already
   normalized to axis probability mass. Validation reports the ordinary mean
   loss over all feasible validation pairs.
4. Save best and resumable checkpoints every epoch. Stop at 100 epochs or after
   15 epochs without validation improvement. Four validation projections per
   epoch report failures explicitly without discarding the target-loss result.

This first phase is teacher pretraining: the targets are projected heuristic
meshes, not waveform predictions. Repair loss is disabled for this initialization
phase to avoid repeatedly solving mesh optimization for every training batch;
all teacher targets themselves satisfy the legal-mesh constraints. Physical
accuracy still requires the next real-FDTD candidate-search/distillation and
validation-selection phase described in the implementation plan. This launcher
stops after pretraining and does not claim physics improvement from target loss.

The 200 IID test scenes are excluded from teacher targets and optimization.
An initial mesh-feasibility probe inadvertently selected the first four records
in the sorted manifest (IID); it inspected geometry/meshing only, not reference
waveforms or model losses, and did not change the predeclared budget list.

## Running and resuming

```bash
.venv/bin/python scripts/start_training_v6.py \
  --campaign artifacts/reference_v6_2000 \
  --output artifacts/training_v6_2000 --workers 12 --gpu 0
```

The launcher runs detached. The same command resumes target preparation or the
saved optimizer under the same source/configuration identity. Source files,
dependency lockfile, native-binary hashes and dirty-tree provenance are recorded;
the source snapshot makes the current uncommitted implementation recoverable.

- `stage.json`: current workflow stage.
- `reference_audit.json`: verified split counts and reference locations.
- `teacher/progress.json`: completed, feasible and failed target pairs.
- `teacher/targets.json`: ordered samples, failures and target identity.
- `pretraining/training.json`: epoch losses, early stopping and projection failures.
- `pretraining/best.pt` and `pretraining/resume.pt`: model and optimizer checkpoints.
- `workflow.log`: preparation and epoch output; `failure.json` records any failure.

Validation before launch: 10 training tests passed, including parallel target
equivalence/resume binding, checkpoint resume and reported projection failures.
A real v6 training raster completed three CUDA AdamW updates at width 16 and
batch 32 with finite loss/gradients, using 1,186 MiB peak allocated memory.

During preparation, only 3 of the first 994 training scenes were feasible at
budget 32, versus 948 at 48 and 992 at 64. The initially configured inverse-count
budget sampler would therefore have repeated those three scenes excessively.
The stopped workflow retained 4,484 completed records and was migrated to uniform
sampling before the first CNN epoch. On user direction, budget 32 was then removed
and replaced with 128. Cached 48/64/96 records are reused; the 32-grid feasibility
records remain archived in the cache but are excluded from target assembly and
training.
