# v5 reference campaign

This document records the implemented v5 campaign contract. The generator and
campaign modules are the source of truth:

- [`src/fdtdmesh/data/generate_v5.py`](../src/fdtdmesh/data/generate_v5.py)
- [`src/fdtdmesh/data/campaign.py`](../src/fdtdmesh/data/campaign.py)

Reference data have not yet been generated. The campaign target is 1,024 accepted
training scenes, 128 validation scenes and 128 test scenes, for 1,280 accepted
scenes total. Each split is filled by replacement attempts across four topology
strata: separated, contact, overlap and nested. Contact, overlap and nested cases
use curated rectangle pairs; tangency and general junctions are not claimed as
implemented generator strata. The requested primitive count is 8–16; the realized
count is recorded for every scene. A scene is rejected when finite-body composition
leaves hidden or tiny components, thin necks, a normalized primitive span below
0.125, a visible core radius below 3 pixels at 128×128, or a visible fraction
below 0.25. Every scene contains at least one finite PEC body. An optional 0–2
axis-aligned zero-width PEC lines may be included; lines are exempt from area/core
checks but must span at least 0.125 of the domain and render at least 8 pixels at
128.

Materials vary `epsilon_r` on a stratified logarithmic 1–30 range and sample
electric loss by carrier ratio `1e-4` to `1`, with 25% exactly zero. `mu_r` is
fixed at 1 and `sigma_h` is fixed at 0 in this campaign. The model retains the
`sigma_h/(2*pi*f_max*MU0)` channel, which is therefore zero. PEC bodies are a
required category, with 10% categorical body sampling where applicable. Magnetic
loss solver capability is retained for separate tests; the campaign configuration
uses `magnetic_loss=False`, so it is inactive and not
production validated here. The SI H-conductivity convention is described for
conceptual context in [Meep's materials documentation](https://meep.readthedocs.io/en/latest/Materials/);
the campaign does not generate magnetic-loss materials.

The width-16 model has ten raw input channels (128,578 parameters). Checkpoint format 3 records
the channel order and normalization and rejects format-2 checkpoints. Native
magnetic damping remains a tested solver capability, but is inactive in campaign
scenes and not production validated.

All source/port and receiver `x`/`y` coordinates are hard anchors when v5 sets
`SceneSpec.anchor_probes=true`; the legacy default remains `false` and is omitted
from legacy hashes. PEC-line positions and endpoints are also mandatory anchors.
These coordinates use the normalized 1/128 or 1/64 lattice, compatible with the
128-grid reference doublings.

## Reference acceptance

The float64 reference sweep considers levels 128, 256, 512, 1024, 2048 and 4096,
with 1024 as the minimum accepted level. It requires two consecutive refinements
with waveform and spectrum relative errors at most 2%, a tail error at most 1%,
up to six duration doublings, and an extended spatial check after duration
extension. Resolution requirements are 16 cells per wavelength and 4 cells per
attenuation length. A scene attempt is bounded by 20 trillion cell updates per
grid/window, 6 GB estimated field storage, 1 GB receiver history, 262145
timepoints, 32769 frequency samples, 6 hours. Separately, a 20,000-candidate safety ceiling per split/lane halts
the campaign as incomplete if a quota still cannot be met.

Every rejected scene, failed attempt and raw reference attempt is retained with
its status and level progress. An unexpected solver error halts its worker. The
campaign publishes `accepted_manifest.json` and `complete.json` only after every
split/topology quota and accepted artifact check succeeds.

Run the campaign from the project root:

```bash
.venv/bin/python -m fdtdmesh.data.campaign --output artifacts/reference_v5 --gpus 0 1 2 3 --train 1024 --validation 128 --test 128
```

Resumption is bound to the saved campaign configuration, generator/source/native
provenance and reference settings. Resume an interrupted run only when those
identities match; a changed contract requires a new output directory. Use
[`scripts/qualify_v5.py`](../scripts/qualify_v5.py) for the development gallery
and audit before the production sweep.


For a detached Linux launch that survives terminal closure:

```bash
.venv/bin/python scripts/start_reference_campaign.py --output artifacts/reference_v5 --gpus 0 1 2 3 --train 1024 --validation 128 --test 128
```

The coordinator lock rejects duplicate runs. `progress.json` reports aggregate
coverage; `lane*/active.json` identifies current scenes. Per-attempt
`references/<scene>/progress.jsonl` records refinement and duration history.
Only a satisfied quota plus successful final integrity checks creates `complete.json`.
A restart repeats at most the unfinished scene on each worker; it does not
checkpoint CUDA fields or CPML history mid-scene.

## Sampled material mapping

New quota-campaign launches default to `--material-averaging sampled`, with
8×8 initial, 32×32 maximum samples and 1e-3 successive-parameter tolerance.
The existing ten-scene pilot remains point sampled. The method is part of the
saved reference contract and requires a separate output directory when changed.
See [sampled material averaging](sampled_material_averaging.md) for the PEC
exclusions, API, sampling limitations, and measured convergence comparison.
The measured improvement in dielectric-only controls does not establish that
PEC-rich campaign scenes converge within the current limits.
