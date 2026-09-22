# High-contrast dielectric reference escalation

This stage uses the validated float64 CUDA backend to revisit the deferred
high-contrast single-cylinder scenes. Each run stores source-normalized complex
far fields, analytic fields, derived scattering widths, the exact configuration,
and runtime diagnostics. The runner resumes only after its fingerprint and saved
arrays pass integrity checks.

The individual acceptance gates are maximum complex angular L2 error below 2%,
weighted phase RMS below 1.5 degrees, last-window field tail below `1e-5`, and
analytic-series change below `1e-10`. An accepted scene must also change by less
than 0.5% under independent duration, material-quadrature, and NF2FF-contour probes.
The variation is the angular complex-field L2 change, normalized by the analytic
field norm and maximized over 0.8, 1.0, and 1.2 GHz.

| Scene | Accepted base | Complex error | Phase RMS | Tail/peak | Duration change | Quadrature change | Contour change | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| eps_r=12, sigma_e=0 S/m | 512², 400 ns | 1.668% | 0.842° | 7.20e-6 | 0.0312% | 0.0204% | 0.0358% | Accepted |
| eps_r=30, sigma_e=0.2 S/m | 512², 50 ns | 0.885% | 0.339° | 1.89e-7 | 0.0000010% | 0.0113% | 0.0340% | Accepted |
| eps_r=30, sigma_e=0 S/m | None | 3.678% at base | 2.082° | 7.92e-2 | 1.979% | — | 1.896% spatial | Nonconverged; skipped |

The lossless eps_r=30 scene reached the declared bounded pilot limit: a 512² run
extended to 400 ns and an independent 768² spatial probe run for 100 ns. The long
run still had tail/peak 0.0356, and its complex field changed by 1.979% from the
100 ns result. The 768² field changed by 1.896% from the 512² result. It is retained
as a failed high-Q example and is not promoted to a training reference. This hard
limit bounds this qualification stage; it is not a claim that the physical problem
cannot converge at still greater cost.

Run or resume one attempt with, for example:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dielectric_reference_attempt.py \
  --scene eps12_small --cells 512 --duration-ns 400 --device cuda:0
```

Regenerate the deterministic report and plot after all required attempts exist:

```bash
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/summarize_dielectric_escalation.py
```

Local outputs are under `runs/dielectric_reference_escalation/`; `report.json`
contains all per-frequency variation values and `qualification.png` shows the
individual accuracy and settling gates.
