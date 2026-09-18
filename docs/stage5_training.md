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

## Available-data continuation

The larger local reference run was stopped for migration to external servers after
172/704 training-scene decisions. It retained 79 converged references and 93
nonconverged scenes with no runtime failures. Before that run, five promising
single-pass cases were refined through 2048 and all five achieved the required
second consecutive pass. Four resource-limited cases were retried with a
256-billion-update cap: one converged at 1024 and three completed as explicit
nonconverged rejects. The acceptance criteria were unchanged.

Training and validation continued locally on the complete pilot corpus. A second
candidate search used the current Stage 5 checkpoint, the Stage 4 checkpoint,
uniform and heuristic baselines, three mixtures, and two seeded perturbations. It
again produced 72 targets (60 train and 12 validation). Winner counts were:

| Candidate | Targets |
|---|---:|
| uniform | 20 |
| uniform + Stage 5 CNN | 16 |
| uniform + heuristic | 11 |
| Stage 5 CNN | 8 |
| Stage 4 CNN | 4 |
| heuristic | 4 |
| Stage 5 perturbation 0 | 3 |
| Stage 5 perturbation 1 | 3 |
| heuristic + Stage 5 CNN | 3 |

Two continuations started from the first Stage 5 checkpoint. The 50% physics blend
selected epoch 11 and stopped at epoch 21; its best validation CDF loss was
5.711e-5. The stronger 80% blend selected epoch 7 and stopped at epoch 17; its best
CDF loss was worse at 6.512e-5. Both used a 60-epoch cap and ten-epoch patience.

Checkpoint selection used actual validation FDTD measurements rather than CDF
loss. Across 12 validation scene-budget pairs, the 80% model reduced median maximum
EM error from 5.880% to 4.934% and median cell updates from 33,211,392 to 30,456,320
relative to the 50% model. Its error-plus-cost objective was 0.07164 versus 0.08179,
so the 80% checkpoint was selected.

The selected checkpoint completed all 32 untouched IID scene-budget evaluations:

| Strategy | Successful | Median max EM error | Mean max EM error | Median cell updates |
|---|---:|---:|---:|---:|
| uniform | 32 | 4.680% | 5.502% | 24,367,104 |
| heuristic | 32 | 4.223% | 4.801% | 30,864,384 |
| prior Stage 5 | 32 | 4.328% | 5.173% | 27,440,640 |
| selected 80% model | 32 | **4.068%** | 4.902% | 26,686,976 |

The selected model has lower error than uniform in 20/32 pairs and than the
heuristic in 17/32. It uses less work than the heuristic in 29/32 pairs. Against
the prior Stage 5 model, aggregate median and mean errors and median work all
improve, although paired error is lower in 15/32; the median paired error ratio is
1.002 because the aggregate median and paired-ratio statistics summarize different
distributions. The summary figure is
`artifacts/stage5_available/stage5_available_summary.png`.

Continuation SHA-256 values:

- 50% physics targets: `88f2ae4cf834880a508d6a5ef432057477beece775b45ca9a5ab70fa0c93962c`
- 80% physics targets: `d35707547051586e3d5149190a14a5813ef7cbfad3970073e747a6e0d25432e3`
- 50% checkpoint: `111b566c9f22e1f565bf0433aec8d7bc63ed6137dde52a2331b4698840fbcc03`
- selected 80% checkpoint: `b91bd56e06162dc61c1350bc67c9c823c7515cc1948476004fb23685c0209f0c`

## Low-budget continuation

The pilot references were reused without changing their manifest identity or
recomputing converged waveforms. Teacher generation, candidate search, and evaluation
now accept an explicit list of square budgets. The run used 32x32, 48x48, 64x64,
and 96x96. Teacher projection produced 308 feasible pairs across all selected scenes
and recorded 12 infeasible pairs. Physics search used the 36 converged train and
validation references and produced 142 targets: 34 at 32x32 and 36 at each other
budget. The two missing 32x32 pairs remain explicit geometric infeasibilities.

The 80% physics continuation started from the previously selected checkpoint. Its
best validation CDF loss was 1.803e-4 at epoch 24; training stopped at epoch 34 after
ten stale epochs. Validation FDTD improved aggregate median maximum EM error from
9.723% to 7.300% and mean error from 23.764% to 21.845%, winning 14/24 pairs.

Held-out IID results span 63 feasible scene-budget pairs:

| Budget | Feasible | Low-budget median | Prior Stage 5 median | Low-budget mean | Prior Stage 5 mean |
|---:|---:|---:|---:|---:|---:|
| 32x32 | 15 | **27.736%** | 30.141% | 31.428% | **28.549%** |
| 48x48 | 16 | **10.018%** | 10.455% | 11.163% | **10.814%** |
| 64x64 | 16 | **5.234%** | 5.597% | 6.404% | **5.713%** |
| 96x96 | 16 | 3.455% | **3.394%** | **3.587%** | 4.091% |
| all | 63 | 7.507% | **7.044%** | 12.855% | **12.034%** |

The continuation wins 35/63 paired comparisons and improves the median in every new
low-budget stratum, but its aggregate held-out median and mean are worse. It is
therefore retained as an experimental low-budget checkpoint and does not replace the
current default. The comparison and mesh figures are
`artifacts/stage5_low_budget/low_budget_test_comparison.png` and
`artifacts/stage5_low_budget/low_budget_cnn_meshes.png`.

Low-budget SHA-256 values:

- heuristic targets: `593c4ada4b0c0ffaf8eb54433608b0d105b51048775d4401c6fdbfc8ce39904a`
- 50% search targets: `3caee15225502948ebe171452ed88fb82b4f44be067e43273c20aba5aa6ba677`
- 80% blended targets: `d7fac498586861db3c8e33e37110252a41bf1bfdcced2219fbfde296043697bf`
- experimental checkpoint: `dff6a46f631e807824466dec45c5a96383afbef389050908bb5a959df7154c4e`

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
