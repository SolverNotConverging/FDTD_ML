# Stage 5: first physics-selected dataset and distillation

This record covers the first substantial Stage 5 cycle. It uses strict converged
uniform references, real CUDA candidate simulations, Pareto-aware target selection,
and a distilled checkpoint initialized from the Stage 4 imitation model.

## Reference corpus

The generator-v3 manifest has dataset ID
`f17978780a329ec23e184b889440c52e5904e737d122d3a6d86bc5ae58abf7d7`
and contains 64 train, 16 validation, and 32 untouched IID test scenes. The full
reference sweep uses uniform 64, 128, 256, 512, and 1024 grids; 2% maximum receiver
waveform and complex-spectrum error; two consecutive passes; a 1% late-time gate;
and up to three duration doublings.

Of 112 scenes, 52 converge, 56 remain spatially nonconverged, and four fail explicit
resource checks. Coverage is 30/64 train, 6/16 validation, and 16/32 test. Accepted
levels are 256 for 14 scenes, 512 for 11, and 1024 for 27. Every accepted scene has
an NPZ waveform/mesh record. The scene runs consumed 1.00 summed GPU-job hours.

## Physics targets

For both 64x64 and 96x96 budgets, each accepted train/validation scene evaluates
uniform, heuristic, Stage 4 CNN, three mixtures, and two seeded CNN perturbations.
Candidates use the accepted physical duration and are rejected if failed, nonfinite,
or unsettled. Selection retains the error/cost Pareto frontier and minimizes
`max(waveform_l2_max, spectrum_l2_max) + 0.02 * normalized_cell_updates`.

The resulting dataset has **72 targets**: 60 train and 12 validation. All target
arrays are finite and their stored SHA-256 matches. Winner counts are:

| Candidate | Targets |
|---|---:|
| uniform | 23 |
| uniform + CNN | 11 |
| uniform + heuristic | 11 |
| CNN | 8 |
| heuristic | 8 |
| heuristic + CNN | 6 |
| CNN perturbation 1 | 4 |
| CNN perturbation 0 | 1 |

Median selected EM error is 3.788%; median normalized work is 1.004. The distilled
target blends the selected projected density 50/50 with the Stage 4 teacher prior.
This first-cycle prior prevents an abrupt target shift while preserving measured
physics variation.

## Training and held-out evaluation

The width-16 ResU-Net starts from the Stage 4 checkpoint and trains for 20 epochs
with AdamW, batch size four, learning rate 3e-4, and repair weight 0.05 every fourth
batch. Epoch 16 is selected. Relative to the initial checkpoint, physics-target CDF
loss improves from 5.459e-5 to 2.517e-5 on train and from 1.586e-4 to 1.161e-4 on
validation. Training starts from a clean Git commit and takes 144 seconds on the
RTX 4070 Laptop GPU.

The untouched IID evaluation uses all 16 converged test scenes and both budgets:

| Strategy | Successful | Median max EM error | Mean max EM error | Median cell updates |
|---|---:|---:|---:|---:|
| uniform | 32 | 4.680% | 5.502% | 24,367,104 |
| heuristic | 32 | 4.223% | 4.801% | 30,864,384 |
| Stage 4 imitation | 32 | 4.690% | 5.044% | 27,765,760 |
| Stage 5 distilled | 32 | 4.328% | 5.173% | 27,440,640 |

Stage 5 has lower error than Stage 4 in 17/32 paired cases and lower work in 20/32.
The median paired ratios are 0.989 error and 0.976 work. Against uniform, Stage 5
has lower error in 21/32 cases with median paired error ratio 0.925, at a median work
ratio of 1.122. Against the heuristic it has lower error in 17/32 and lower work in
28/32, with median paired ratios 0.969 error and 0.864 work. Aggregate means remain
sensitive to outliers; the Stage 5 mean is slightly worse than Stage 4 even though
its median improves.

## Reproduction

```powershell
.venv\Scripts\python.exe -m fdtdmesh.training targets --manifest artifacts/reference_pilot/manifest.json --output artifacts/reference_pilot/teacher_targets.npz --splits train validation
.venv\Scripts\python.exe -m fdtdmesh.physics search --manifest artifacts/reference_pilot/manifest.json --teacher-targets artifacts/reference_pilot/teacher_targets.npz --checkpoint artifacts/stage4/training/best.pt --references artifacts/reference_pilot/references --output artifacts/stage5/search --splits train validation --beta 0.02 --physics-weight 0.5 --perturbations 2 --device cuda
.venv\Scripts\python.exe -m fdtdmesh.physics distill --manifest artifacts/reference_pilot/manifest.json --targets artifacts/stage5/search/physics_targets.npz --initial-checkpoint artifacts/stage4/training/best.pt --output artifacts/stage5/training --epochs 20 --batch-size 4 --learning-rate 0.0003 --device cuda
.venv\Scripts\python.exe -m fdtdmesh.physics evaluate --manifest artifacts/reference_pilot/manifest.json --checkpoint artifacts/stage5/training/best.pt --baseline-checkpoint artifacts/stage4/training/best.pt --references artifacts/reference_pilot/references --output artifacts/stage5/evaluation --split test_iid --device cuda
.venv\Scripts\python.exe examples/plot_stage5.py --training artifacts/stage5/training/training.json --evaluation artifacts/stage5/evaluation/report.json --targets artifacts/stage5/search/physics_targets.json --output artifacts/stage5/stage5_summary.png
```

Key SHA-256 values:

- physics targets: `f2735a29709154d6b762f441073d43a363bfc43fe27c0d8986017012538f7c78`
- selected checkpoint: `96a806efef6e26aae241fa806644ed755652227aad2b8fdaf9e0a81466e41afb`
- exact resume state: `03bf1fd39a9d70515f0f6debf32e4304cd10958858f88c787ceca836e2098301`

The checkpoint records clean training commit `8fde46689804d5c5eba3dc6dda4ba199e42f0a8f`.
Generated artifacts remain ignored because reference waveforms, candidate histories,
and checkpoints are large; the manifest identity, commands, configurations, hashes,
and measured results are committed here.

Final verification: **131 tests passed with no skips**, including CUDA, in 16.80
seconds. Ruff lint/format and locked offline `uv sync` also pass. The summary figure
was rendered and visually inspected.
