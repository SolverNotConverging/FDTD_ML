# Generator v3: dielectric distribution and spatial probe coverage

Implemented 2026-09-18, package 0.4.1. Scene schema v1 and checkpoint format v2 remain
compatible. New manifests store generator version 3 and every generation setting.
The additional mixture draw changes seed realizations; historical v2 generation is
reproducible at Git commit `00db055`. Existing manifests retain their exact scenes.

## Dielectric sampling

Each ordinary dielectric object independently draws a mixture component:

| Component | Default probability | Distribution of dk = epsilon_r |
|---|---:|---|
| Core | 85% | Log-uniform 1–10 |
| Tail | 15% | Log-uniform 10–30 |

The density within each component is proportional to 1/dk, giving decreasing
frequency as dk increases. This is a bounded mixture, not an unbounded statistical
heavy tail. The probability is a configurable initial design choice, not a measured
real-world prevalence. Packing, resolution and split rejection condition the accepted
dataset; small samples will not contain exactly 85% core objects. Counts are per
dielectric object, excluding PEC and background vacuum.

`GenerationConfig` exposes `epsilon_min`, `epsilon_core_max`, `epsilon_max` and
`epsilon_core_probability`. Cutoffs must be ordered and finite; probabilities must
lie in [0,1]. The CLI exposes `--dk-core-max` and `--dk-core-probability`.
Material-OOD deliberately uses log-uniform 36–120, derived from the configured
maximum; it does not follow the ordinary mixture. Conductivity and all other broad
geometry distributions retain the [v2 contract](dataset_v2.md).

## Spatial probes

The existing v2 spatial policy already randomizes the source and each of three
receivers. Each proposes independent x/Lx and y/Ly uniformly over [0.19,0.81].
Rejection sampling enforces a 0.04 normalized-axis margin around geometry group
bounding boxes and a minimum normalized Euclidean separation of 0.08 between probes.
Thus accepted positions depend on geometry and on earlier probes; they are not
uniform over the entire domain. PML collars remain excluded. Every role can appear
in every quadrant, with no fixed coordinate or prescribed left-to-right ordering.
This removes the fixed-source-location shortcut, without proving learned generalization.

## What convergence means

Spatial reference acceptance uses both receiver observables, on common physical
times and over the same duration:

1. Relative L2 error of the full time-domain waveform <=2% for every receiver.
2. Relative L2 error of the complex, Hann-windowed DFT <=2% for every receiver,
   aggregating over sampled frequencies between f_min and f_max.

Both must pass for two consecutive grid refinements. Complex differences include
amplitude and phase; phase RMS is reported separately and is not another acceptance
threshold. The spectral criterion is a norm over the band, not a per-frequency 2%
bound. Quiet-signal denominator floors remain active. DFTs are calculated from the
sampled receiver time traces, rather than a separate frequency-domain solver.

Time-window acceptance is a separate time-domain test: after the source pulse ends,
each receiver's final-20%-window RMS divided by its full-trace peak must be <=1%.
Otherwise the duration doubles and spatial refinement restarts, up to two extensions
by default. The accepted duration is used for all candidate meshes. Receiver settling
does not prove every unobserved field mode has decayed. These criteria are unchanged
by the generator-v3 update.

## Reproduction and evidence

```powershell
.venv\Scripts\python.exe -m fdtdmesh.data generate --output artifacts/stage3_v3/manifest.json --per-split 32 --seed 2026 --dk-core-max 10 --dk-core-probability 0.85
.venv\Scripts\python.exe examples/plot_dataset_variation.py --manifest artifacts/stage3_v3/manifest.json --output artifacts/stage3_v3
```

The 256-scene manifest has dataset identity:

```text
10c7b9dcaf1c9b47bf723d33c39bf354d659c9b7b63500b84446a8d4ebc1266f
```

Across 96 train/validation/IID scenes there are 267 dielectric objects, of which
237 (88.76%) have dk <=10. Values span 1.0201–29.1278. Each probe role spans more
than 0.60 of each normalized axis. The source covers x/Lx=0.1918–0.8093 and
y/Ly=0.1906–0.8097. Exact per-receiver ranges are saved in `diversity_summary.json`;
`probes_and_permittivity.png` was visually inspected. These generated artifacts are
local and ignored by Git; the scripts and measured record are versioned.

All **109 tests pass with no skips** (16.35 seconds), including three real-CUDA mesh
strategies. Coverage includes deterministic scene generation, mixture frequency and
within-component distribution, probability endpoints, invalid settings, high-dk OOD,
per-role spatial coverage and separation, and the existing convergence tests.
Ruff lint/format and offline locked uv synchronization pass. No full v3 reference
sweep or CNN training was performed; earlier v2 numerical results are historical.
