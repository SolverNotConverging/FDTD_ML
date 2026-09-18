# Converged reference dataset workflow

The first physics-target dataset uses 112 deterministic generator-v3 scenes:

| Split | Scenes | Use |
|---|---:|---|
| `train` | 64 | Physics-target search and distillation |
| `validation` | 16 | Model and search selection |
| `test_iid` | 32 | Untouched final IID evaluation |

Generate the manifest with:

```powershell
.venv\Scripts\python.exe -m fdtdmesh.data reference-manifest `
  --output artifacts/reference_pilot/manifest.json `
  --seed 2026 --train 64 --validation 16 --test 32
```

The recorded manifest has dataset ID
`f17978780a329ec23e184b889440c52e5904e737d122d3a6d86bc5ae58abf7d7`.
It contains exactly 64/16/32 scenes, 1–8 geometry primitives per scene, dielectric
permittivity 1.020–29.128, conductivity 0–9.429 S/m, and source coordinates spanning
approximately 0.19–0.81 on both normalized axes. The manifest is an ignored generated
artifact; its seed, configuration, scene records, hashes, and provenance make it
reproducible.

Generate references with:

```powershell
.venv\Scripts\python.exe -m fdtdmesh.data references `
  --manifest artifacts/reference_pilot/manifest.json `
  --output artifacts/reference_pilot/references `
  --splits train validation test_iid `
  --levels 64 128 256 512 1024 `
  --tolerance 0.02 --consecutive-passes 2 `
  --tail-tolerance 0.01 --duration-extensions 3
```

The output is resumable. `run.json` binds the directory to the dataset ID, selected
splits, limits, mesh policy, and full convergence configuration. Repeating the exact
command skips every scene with a committed `reference.json`. A changed dataset,
selection, or convergence configuration is rejected instead of mixing results.
`report.json` and `summary.md` are rewritten after every scene, so an interruption
loses at most the active scene.

Each scene chooses its own duration and spatial level. The base duration already
accounts for pulse width, domain transit time, and material permittivity. If the last
20% receiver RMS exceeds 1% of peak after the pulse ends, duration doubles and the
spatial sequence restarts. This can happen up to three times in this production
command. Once the time window settles, uniform 64, 128, 256, 512, and 1024 grids are
tested until both waveform and complex-DFT relative errors are at most 2% for two
successive refinements. Fast scenes stop as soon as both passes succeed; difficult
scenes consume the larger duration or finer grids they need.

Only a `converged` status is valid ground truth. `nonconverged`, `time_unsettled`, and
`failed` scenes remain in the coverage report and are never silently promoted. The
latest waveform and mesh are retained for diagnosis, but `reference.json` controls
acceptance. Material coefficients continue to use direct Yee-location point sampling;
this workflow does not add subpixel or effective-medium averaging.

The solver currently restarts from zero for each longer duration because CPML history
state is not exposed for continuation. Restarting costs extra work but ensures every
spatial level is compared over the same physical window. Full receiver histories also
remain proportional to simulation length; bounded recording is future work.
