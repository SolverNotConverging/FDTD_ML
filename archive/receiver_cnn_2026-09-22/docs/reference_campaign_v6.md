# v6 reference campaign: rectangular and thin-wire PEC

The current proof-of-concept dataset uses generator v6. Legacy v4/v5 generators
and their saved scenes remain available for reproducibility; v6 scene IDs,
streams, and dataset hashes are distinct.

## Geometry and anchors

- Dielectric bodies retain rectangle, circle, triangle and polygon variation,
  with separated/contact/overlap/nested dielectric-pair strata and the existing
  visible-feature checks. Finite body counts remain 8–16 before visibility
  rejection; the accepted distribution can favor fewer bodies.
- Epsilon_r varies up to 30 and sigma_e varies; mu_r is fixed to 1 and sigma_h
  to 0. V6 rejects configurations that enable magnetic variation.
- Finite PEC bodies are axis-aligned rectangles only. Each requires **four mesh
  lines**: x=x_min, x=x_max, y=y_min, y=y_max.
- Thin PEC wires are zero-width axis-aligned lines. Each requires **three mesh
  lines**: one on the wire's fixed coordinate and two through its endpoints on
  the other axis. These are not three parallel lines surrounding the wire.
- The global anchor sets deduplicate coordinates shared by multiple objects.
  Every source and receiver retains both of its coordinate anchors.
- Dielectrics are inserted first, then PEC rectangles, then wires. This prevents
  later dielectric circles/polygons from cutting nonrectangular holes into PEC.
  The scene policy rejects nonrectangular PEC and later dielectric overrides.

PEC rectangle faces and wire coordinates are selected on a **1/64-domain
lattice** when the continuous scene is generated. This defines the geometry;
no solver-dependent snapping occurs later. Probe coordinates use the 1/128
lattice. All 128→2048 campaign reference grids therefore contain the exact
mandatory anchors. This avoids per-scene nonuniform reference meshing and
anchor-induced tiny timesteps, while candidate/CNN meshes still enforce the
same coordinates through projection. Some small candidate budgets can remain
infeasible under exact anchors and 1.4 grading; anchors are never silently dropped.

`SceneSpec.pec_policy="rectangles_and_wires"` serializes the geometry/anchor
contract. Historical scenes default to `"legacy"` and preserve their old content
hashes. The generic solver retains legacy geometry support for old experiments.

## Physics and acceptance

Sampled Ez dual-cell averaging is enabled: 8×8 initial, 32×32 maximum samples,
1e-3 successive-parameter tolerance, with PEC exclusions unchanged. See
[sampled averaging](sampled_material_averaging.md).

The active reference policy uses float64 levels 128/256/512/1024/2048.
Acceptance remains at 1024 or finer, with two consecutive waveform AND spectrum
differences at most 2%, a settled tail at most 1%, and material wavelength and
attenuation resolution checks. A scene still spatially nonconverged after 2048
is recorded as nonconverged and immediately replaced. Physically unsettled tails
may still receive up to six duration doublings. Existing resource and six-hour
per-scene limits remain. Quota generation replaces exhausted candidates until
1600 training + 200 validation + 200 IID references are accepted, or reports an
explicit safety-limit failure. It never counts failed scenes as valid references.

## Pilot and launch

The previous mixed-PEC v5 pilot was stopped as superseded. Its partial decisions,
progress, timing-contention metadata and stop record remain under
`artifacts/reference_v5_pilot10/`.

The v6 ten-scene manifest is precomputed at `artifacts/v6_validation/manifest.json`.
All ten scenes passed feature and mandatory-anchor checks on reference grids
128, 256 and 1024. Three of the ten contain wires. This is a newly generated
v6 batch, not a claim that its scene IDs/geometries match v5.

```bash
.venv/bin/python scripts/pilot_reference_v6.py \
  --count 4 --manifest artifacts/v6_validation/manifest.json \
  --output artifacts/reference_v6_pilot4
```

At the user’s request, the pilot runs only the first four candidates on four GPUs. Every status is retained.
Its report distinguishes per-scene wall time from aggregate throughput; when a
precomputed manifest is supplied, geometry-generation time is excluded.
Measured results are recorded below, separately from preflight evidence.

The full 2,000-reference quota campaign was launched on 2026-09-19:

```bash
.venv/bin/python scripts/start_reference_campaign.py \
  --output artifacts/reference_v6_2000 --gpus 0 1 2 3 \
  --train 1600 --validation 200 --test 200 \
  --max-reference-level 2048 --no-extend-nonconverged
```

The campaign runs detached; progress is recorded in
`artifacts/reference_v6_2000/progress.json`, with coordinator and lane logs in
the same directory. Only converged cases count toward the 2,000 total.
The user will follow up when generation finishes; no completion monitor is scheduled.

The campaign was stopped and migrated to this ceiling after 552 decisions. The
17 references that had required 4096 were reclassified as nonconverged, leaving
530 accepted references at migration time. Four interrupted attempts were
archived and restarted under the new policy. The complete pre-migration config,
progress, launch metadata and interrupted outputs are preserved in
`artifacts/reference_v6_2000/policy_migration_max_2048/`.

New campaigns default to material averaging. Changed source, generator,
geometry/anchor policy or reference settings require a new output identity.

## Four-scene pilot result (2026-09-19)

All four requested scenes converged at 1024×1024, with no duration extensions.
Mean per-scene wall time: **57.69 s**, including setup and all refinement levels,
excluding precomputed geometry generation. The four-GPU batch elapsed **69.09 s**
(17.27 s per accepted scene in aggregate). Reference-worker mean: **54.88 s**.

| Scene | Family | Time (s) | Final waveform difference (%) | Final spectrum difference (%) |
|---|---|---:|---:|---:|
| train-v6-000000 | separated | 69.09 | 0.092 | 0.088 |
| train-v6-000001 | contact | 52.74 | 0.182 | 0.185 |
| train-v6-000002 | overlap | 61.22 | 0.052 | 0.052 |
| train-v6-000003 | nested | 47.72 | 0.345 | 0.344 |

All cases passed consecutive-refinement, decay and resolution gates. These are
new v6 geometries; this is not a paired causal comparison against the v5 pilot.
Validation: full suite **167 passed** (`artifacts/v6_validation/full_tests.log`);
pilot/plot/report scripts pass Ruff. No new training was launched during the pilot.

Artifacts: `artifacts/reference_v6_pilot4/{summary.json,results.md,pilot_results.png}`.
Four-scene geometry gallery and wire-anchor close-up:
`artifacts/v6_validation/{pec_geometry_gallery.png,pec_anchor_mesh.png}`.
