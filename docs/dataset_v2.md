# Generator v2: broader geometry and longer resonance windows

Historical generator-v2 record. Current generation uses the [v3 dk mixture](dataset_v3.md).
To reproduce v2 from seeds, use Git commit `00db055`; existing saved manifests remain
readable. The commands below describe the v2 checkout, not current defaults.

Implemented 2026-09-18, project version 0.4.0. Scene/manifest schema v1 and checkpoint
format v2 remain compatible. Generator version 2 and its full configuration are
stored in new manifests. `fdtdmesh.data.generate_v1` preserves the original generator
for historical reproduction; old manifests remain readable. Historical baseline
results in `stage3_validation.md` do not describe this broader distribution.

## Randomization contract

`GenerationConfig` controls ranges and probabilities. Seeds reproduce object counts,
shapes, placement, materials, probes and pulse parameters. Positive ranges use
log-uniform draws to cover small and large scales; locations/angles and integer
counts use uniform draws. Packing, feature-resolution and split checks condition
the accepted distribution, so accepted scenes are not strictly uniform in these
parameters. No accuracy score is used to select generated scenes.

| Parameter | Ordinary training/validation/IID distribution |
|---|---|
| Objects per scene | Random integer 1–8, counting each rectangle of a pair separately |
| Lx | Log-uniform 12–60 mm |
| Ly/Lx | Log-uniform 0.3–3.3 |
| f_max | Log-uniform 8–40 GHz; f_min=0.05 f_max |
| Geometry spans | Log-uniform 3.5–50% of an axis before placement constraints |
| Relative permittivity | Independent log-uniform 1.05–30 for each dielectric object |
| Conductivity | 20% exactly lossless; otherwise independent log-uniform 1e-5–10 S/m |
| PEC | 25% probability for ordinary material-bearing objects; explicit PEC lines also occur |
| Permeability | mu_r=1; magnetic variation is not part of this generator |
| CNN raster | 128x128, maintaining the four-pixel minimum bounding-span/gap rule |
| Candidate budgets | 64x64 and 96x96; fixed eight-cell PML collars per side |

Each ordinary scene randomly mixes axis-aligned rectangles, circles, oriented
triangles and quadrilaterals, horizontal/vertical PEC lines, and two-object
rectangular gap/touching motifs. There is no index-based family or material cycle.
Every dielectric primitive receives a distinct material with independent epsilon
and sigma, including both objects in touching/gap pairs. Shapes are retained in
continuous coordinates and insertion order.

Circles retain circular physical geometry: their diameter must span four pixels
on both axes, which narrows their available radius range at extreme domain aspect
ratios. Triangles/quadrilaterals are affine regular polygons with random orientation
and independently varying bounding dimensions; quadrilaterals are not all rectangles.
PEC line anchors are rounded to a 1/64-domain lattice to remain representable by
uniform reference grids. Different object groups have a four-pixel bounding-box
clearance; contact within an intentional touching pair is allowed. Overlap/nesting
and arbitrary narrow concave notches are outside this generator's current scope.

Source and three receivers are drawn in vacuum with geometry clearance and mutual
separation. The source remains an integrated 1 mA point current; pulse width varies
from 0.8/f_max to 1.5/f_max, carrier from 0.2 to 0.65 f_max, and delay is four pulse
widths. Candidate-grid source checks still reject a stencil that lands on PEC;
continuous clearance is not a guarantee for every possible adaptive mesh.

Held-out distributions remain explicit:

- Dense/compositional: 9–12 objects with default configuration (max_objects+1 through
  max_objects+4). Ordinary training now already includes mixed and touching pairs.
- Geometry: randomly oriented 5–9-vertex polygons, which ordinary scenes exclude.
- Material: every object is dielectric; epsilon_r 36–120 and sigma 20–200 S/m.
  These extend the configured ordinary maxima by fixed factors.
- Scale: Lx 70–120 mm and f_max 50–80 GHz, with the same broad aspect range.
- Budget: 80x80 and 112x112; all variants of one scene retain the same split.

Existing lineage and normalized-mask near-duplicate checks remain active. The mask
policy uses 64x64 occupancy and a conservative 0.5%-domain difference threshold;
it can reject distinct small objects. It is not a complete geometric similarity
proof. Explicit generator limits and rejection sampling may fail for overpacked
user configurations rather than silently shrinking or deleting objects.

## Temporal settling and spatial convergence are separate

The initial duration is the larger of:

```text
24 / f_max
8 * source_width + 4 * domain_diagonal * sqrt(max_object_epsilon_r) / c0
```

The coefficients 24 and 4 are configurable through `GenerationConfig`; this is a
transit-time allowance, not a proof that resonances have decayed. The v2 sample's
initial durations range from 0.697 to 11.803 ns.

At each reference resolution, the evaluator checks the final 20% of each receiver
waveform. Tail RMS divided by that receiver's peak (with the existing absolute
floor) must be <=1%. This is tested only after the source's delay plus four widths
precedes the tail window. If unsettled, the duration doubles and the spatial
refinement sequence restarts from the first level. Two extensions are allowed by
default, giving at most four times the requested initial duration. Persistent
ring-down is labelled `time_unsettled`, distinct from spatial `nonconverged` and
solver/resource `failed`. A continuously driven source cannot pass this pulse
settling gate; use explicit fixed-window evaluation where appropriate.

Uniform reference defaults are now 64,128,256,512,1024 cells per axis. Two consecutive
refinement comparisons must still pass the 2% per-receiver waveform and spectrum
L2 criteria. Longer duration captures more late-time behavior and may expose more
phase error; it does not replace mesh refinement. Receiver-tail checks also cannot
certify that unobserved modes or total field energy have decayed.

The final accepted duration is applied to every candidate. Spatial comparisons
never mix durations. Reports save the original scene, the effective `evaluated_scene.json`,
every duration attempt and its per-level tail metrics. Candidate tail diagnostics
are saved too, without hiding poor candidates from error comparison.

Sampling grows with duration: at least 1,025 temporal samples and 16 samples per
highest-frequency period, and spectral spacing no larger than 1/(2T), with at least
12 frequency samples. This avoids extending time by simply stretching an unchanged
sampling grid or retaining a sparse frequency list. DFT evaluation processes blocks
of 64 frequencies to bound temporary memory. Default limits are 65,537 observation
samples, 8,193 frequencies, 128 billion cell updates per run, 256 MB raw receiver
history and a 1 GB field/coefficient estimate. Explicit limits still apply to longer
runs and can be raised through `EvaluationConfig`; the CLI exposes the work cap.
Receiver-tail acceptance is empirical and does not establish broadband accuracy or
capture every narrow resonance in every scene.

## Use

```powershell
.venv\Scripts\python.exe -m fdtdmesh.data generate --output artifacts/stage3_v2/manifest.json --per-split 16 --seed 2026 --objects 1 8 --aspect 0.3 3.3
.venv\Scripts\python.exe examples/plot_dataset_variation.py --manifest artifacts/stage3_v2/manifest.json
.venv\Scripts\python.exe -m fdtdmesh.data evaluate --manifest artifacts/stage3_v2/manifest.json --output artifacts/stage3_v2/evaluation --split test_iid --limit 4 --demo-cnn
```

For longer runs, use `--duration-multiplier 2 --duration-extensions 3`; for a larger
work allowance use `--max-cell-updates 256000000000`. Adjust the tail criterion with
`--tail-tolerance`. `--fixed-window` explicitly disables the temporal settling gate
while retaining spatial-convergence checks. All settings are recorded. Use a fresh
output directory. For other material/size controls, pass `GenerationConfig` to
`generate_dataset` from Python and record `config.to_dict()` in the manifest.

The demo CNN now uses a raster large enough for the selected scenes. Supplied
checkpoints with a smaller raster than the scene's resolution contract are rejected.
No trained weights or demonstrated generalization improvement are introduced here.

## Measured evidence

The final suite passes **102 tests with no skips** (15.59 seconds), including real
CUDA on the RTX 4070 Laptop GPU. Ruff lint/format and locked uv synchronization pass.

A deterministic 128-scene dataset (16 per split, seed 2026) has content hash:

```text
695ea11116b188c773a49fdaa8cb40ed04fef83a1b0b3c15a3ebdd7ac26192bc
```

Its 48 training/validation/IID scenes contain every object count from 1 to 8.
Observed aspect ratios are 0.331–3.207, unanchored geometry spans 0.03518–0.46606 of
an axis, epsilon_r 1.100–27.866 and sigma 0–9.429 S/m. The material and dense held-out
splits extend these distributions as specified above. `diverse_scenes.png` and
`diversity_distributions.png` were visually inspected; exact statistics are in
`artifacts/stage3_v2/diversity_summary.json`.

The first four IID scenes were run on CUDA through the new reference workflow.
Three exhausted spatial refinement without two consecutive passes; the fourth
hit the work limit after automatically extending its duration. No candidate accuracy
labels were accepted from this harder four-scene sample. Separate real-CUDA smoke
tests verify uniform/heuristic/CNN meshing on a v2 multi-object scene, but do not
claim a converged accuracy reference for that scene.

The resource-limited scene (`test_iid-00002`, seven objects) was rerun with a
256-billion-update cap:

| Attempt | Duration | Finest grid reached | Tail RMS / peak | Outcome |
|---|---:|---:|---:|---|
| Initial | 2.927 ns | 64x64 | 2.344% | Extend duration |
| Extended | 5.854 ns | 1024x1024 | 0.0655% | Temporally settled; spatially nonconverged |

At 64x64 the extended window reduces the tail to 0.1108%. Spatial waveform changes
on successive refinements are 30.37%, 4.82%, 3.67%, and 1.31%; spectral changes are
67.93%, 8.02%, 4.62%, and 1.65%. The final pair passes 2%, but the previous pair
fails, so two consecutive passes are not established. The 1024 run performs
133,973,409,792 cell updates. Full attempts and arrays are retained in
`artifacts/stage3_v2/resonance_extended/`.

Unit/integration coverage includes deterministic broad distributions, independent
materials, count/range/resolution checks, separated vacuum probes, temporal gate
extension and exhaustion, restarting spatial checks, preserving candidate windows,
auto-growing sampling, chunked/direct DFT equivalence, and three real-CUDA baseline
strategies on new geometry. The earlier generator's numerical fixtures remain
pinned to `generate_v1` so historical regressions do not silently change with new
randomization. This expansion supplies harder, more diverse inputs; reliable labels
for them still require convergence, resources and further numerical qualification.
