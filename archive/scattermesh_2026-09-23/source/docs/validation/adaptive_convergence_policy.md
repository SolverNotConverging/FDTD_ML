# Adaptive reference convergence policy

`scripts/run_reference_convergence_policy.py` sits above the restartable multi-GPU
attempt scheduler. It accepts an immutable policy containing strictly increasing
mesh, duration, and material-sampling levels plus a hard complex-field variation
threshold.

For each scene, the controller first runs a base attempt and interprets its
individual gates:

- a settling-tail failure advances simulation duration;
- complex-field or phase failure advances spatial resolution;
- simultaneous failures advance both controls;
- nonfinite fields or analytic-series failure are retained as unrecoverable;
- an unavailable next level produces an explicit hard-limit skip.

Once the base passes, the controller requires four independent attempts: the next
spatial level, the next duration, the next material quadrature level, and a larger
NF2FF contour. Each probe must pass its own individual gates. Its complex field is
compared with the base at every frequency and normalized by the analytic-reference
norm. A failed spatial or contour variation advances mesh resolution; a failed
duration variation advances time; and a failed quadrature variation advances
sampling. Qualification repeats from the promoted base and always reserves another
level for the corresponding independent probe.

The policy writes atomic state, immutable per-wave scheduler plans, per-attempt
records, escalation history, accepted base/probe paths, and explicit skip reasons.
On resume it revalidates source hashes, configuration fingerprints, and finite
native complex arrays for every accepted base and probe. Corruption invalidates the
acceptance and schedules only the affected physical attempt again.

## Conductive epsilon_r=30 pilot

The tracked [pilot policy](../../configs/dielectric_convergence_pilot.json) uses
512/640/768 spatial levels, 50/100 ns duration levels, 24/48 material samples, and
a 0.5% independent-variation gate. It completed in two waves and five attempts:

| Probe | Maximum complex-field change |
|---|---:|
| 512² to 640² | 0.3374% |
| 50 ns to 100 ns | 0.0000010% |
| 24² to 48² material samples | 0.01134% |
| Default to larger NF2FF contour | 0.03396% |

The 512²/50 ns base passed all individual gates and all four variations remained
below 0.5%, so it was accepted without escalation. A second controller invocation
revalidated all five artifacts and performed no GPU work.

The integration test separately forces simultaneous accuracy/settling failures,
verifies mesh/time promotion followed by a hard-limit skip, and corrupts an accepted
spectrum to verify selective regeneration.

Run or resume the pilot with:

```bash
.venv/bin/python scripts/run_reference_convergence_policy.py \
  --policy configs/dielectric_convergence_pilot.json \
  --output runs/dielectric_convergence_pilot/policy \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

This policy currently drives the three analytic dielectric-cylinder scenes exposed
by the single-attempt runner. The scene schema and geometry generator must be
generalized before using the controller for the broader simple/sparse/complex pool.
