# Mesh-distillation pilot

The first M5 pilot used the 52 accepted labels from exact-budget campaign
`simple_factorial_exact_candidate_pilot_6fb40ec9fe0fe154`: 36 training, eight
validation, and eight test examples. Each example retains all six candidate axis
profiles and their soft-Nt physics scores.

The model is a 532,593-parameter residual U-Net with a 128x128 five-channel raster
and ten global conditioning features. Training ran on CPU for 124 epochs and chose
epoch 76 by validation loss.

| Metric | Result |
|---|---:|
| Initial training profile loss | 7.785e-4 |
| Final training profile loss | 1.170e-4 |
| Best validation profile loss | 8.939e-5 |
| Test profile loss | 1.036e-4 |
| Validation improvement over uniform profile | 20.40x |
| Test improvement over uniform profile | 18.11x |
| Maximum grading-repair fraction | 0 |

All validation and test outputs projected directly to exact requested cell counts
within the 3.0 adjacent-cell grading cap. These are distillation metrics, not
scattering results. The checkpoint is promoted only to the learned-mesh physics
pilot, which reruns all 16 held-out condition/budget pairs through CUDA FDTD and
compares their complex far-field plus RCS score with uniform and searched teachers.
