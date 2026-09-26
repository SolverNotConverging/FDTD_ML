# TMz conformal enlarged-cell scattering

Method identifier: `tmz-conformal-ect-v1`. Coordinates are SI metres. The solver
uses Ez at grid nodes, Hx on y edges, and Hy on x edges, with vacuum constants
and PEC Ez constraints. Objects are invariant along z. This is a TMz reduction
with a conservative enlarged-face construction, not a complete 3D ECT port.
The motivating enlarged-cell literature is
[Xiao and Liu, 2008](https://doi.org/10.1109/TAP.2008.916876).

## Geometry and enlargement

Circle intersections are analytic; polygon intersections use straight segments.
Ordered PEC/vacuum primitives support holes and notches. The open length of an
edge is the vacuum part of its associated magnetic face per unit z depth.
Ordinary conformal Faraday updates divide endpoint-E circulation by that length.

If open length `s < full_length/2`, borrow length `b = full_length/2 - s` from
the adjacent complete vacuum face on the vacuum side. Let its length be `d`,
`S = s+b`, and `r = b/d`. With endpoint circulations `f_s`, `f_d`, use

```
g_s = (f_s + r f_d) / S
g_d = r g_s + (1-r) f_d / d
```

The pair is disjoint from all other pairs. It conserves integrated circulation:
`s g_s + d g_d = f_s + f_d`. Its inverse-length matrix is symmetric,
with entries `Q_ss=1/S`, `Q_sd=Q_ds=r/S`,
`Q_dd=(1-r)/d+r²/S`; determinant `(1-r)/(S d)>0` when `0<b<d`.
The physical cut lengths and PEC silhouette are retained. Hx uses the negative
y circulation; Hy uses the positive x circulation.

Reject a candidate if an edge has disconnected open pieces, a vacuum gap between
two PEC endpoints, a primitive wholly hidden within one cell, or a small face
without an unused complete donor. These configurations need more degrees of
freedom or geometry-aware anchors/refinement. A rotated polygon can encounter
an invalid corner alignment even after refinement. Rejection is deliberate;
there is no silent staircase or area-clipping fallback. The checks do not prove
that an arbitrary complex shape is sufficiently resolved for accurate scattering.

## Timestep and stability

Let `B` be oriented edge incidence on non-PEC E nodes, `W` the transverse dual
width of each magnetic edge, and `M` the product of x/y dual widths at E nodes.
Enlarged partners share the same transverse width, so it commutes with Q.

```
K = sum_axes Bᵀ W Q B
M Ë = -c0² K E
Λ_bound = max_i sum_j |K_ij| / M_ii
dt_bound = min(dt_cartesian, 2 / (c0 sqrt(Λ_bound)))
dt_default = 0.9 dt_bound
```

K is symmetric positive semidefinite. The row-sum bound on `M⁻¹K` bounds its
nonnegative eigenvalues, giving the leapfrog limit. This proves a bound for the
**lossless spatial operator**; CPML is additionally checked empirically. It is
not a theorem about arbitrary absorbing-layer configurations or 3D geometries.
The solver rejects explicit dt at or above this bound. The tiny-cut regression
uses a 1e-10 m open segment with a 0.03125 m background cell and 20,003 lossless
updates; its dt remains above half the Cartesian limit.

## Incident wave and monitors

A matching 1D Yee/CPML calculation on the same x lines supplies the +x plane
wave. TFSF corrections are applied algebraically using total/scattered region
masks, so discrete incident cancellation works on graded grids. Source injection
is a Gaussian cosine pulse. Source, TFSF, contour and phase-origin x positions
must be anchored. Geometry and enlarged donors stay clear of TFSF and PML.
See the conventional construction in
[Schneider Chapter 8](https://eecs.wsu.edu/~schneidj/ufdtd/chap8.pdf).

The closed rectangular contour lies in scattered-field vacuum, outside TFSF and
inside PML. H is interpolated to the E-node contour using physical distances;
edge quadrature is trapezoidal with half weights at each side's corners.
`Ht = nx Hy - ny Hx`. Online DFTs use `exp(-i ω t) dt` at the actual times
`t_E=(n+1)dt`, `t_H=(n+1/2)dt`, and float64 accumulators for either field precision.
Equivalent currents are represented by the sufficient pair `(Ez,Ht)`; their
orientation and impedance factors are applied in the far-field kernel.

With the `exp(+iωt)` phasor convention and outgoing Hankel-2 waves,

```
S(θ) = k/(4 E_inc(origin)) ∮ [(u(θ)·n) Ez - η0 Ht]
                                 exp(i k u(θ)·(r-origin)) dl
σ_2D(θ) = 4 |S(θ)|² / k
```

The independently evaluated circular-cylinder series is
`S = -J0(ka)/H0²(ka) - 2 Σ[n>=1] Jn(ka)/Hn²(ka) cos(nθ)`.
The superscript 2 denotes Hankel kind 2, not squaring. Absolute complex
amplitude and width are compared, rather than rescaling each angular curve.
See [Schneider Chapter 14](https://eecs.wsu.edu/~schneidj/ufdtd/chap14.pdf)
for the 2D near-to-far setting.

## Device-controlled execution and convergence

Native CUDA kernels implement H, E, current DFT, convergence reductions and NF2FF.
A [CUDA conditional WHILE graph](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
repeats updates until the device changes its condition. Python never submits
individual timesteps or decides whether another batch should run. NF2FF is a
graph node after the WHILE loop. CUDA Toolkit 13.x is required by the compiled API.

Every `check_interval` updates (default 2048), for each requested frequency:

1. Measure the ds-weighted L2 change of the E/H current DFT since the last check,
   scaling H by η0. Require `delta < atol |Einc| sqrt(perimeter) + rtol norm`.
2. Require the incident normalization DFT to settle within `(atol+rtol)|Einc|`.
3. Require the maximum residual field `max(|E|,η0|H|)` relative to peak incident
   field to fall below `field_tol` across the domain.

The source guard waits until pulse delay plus six pulse widths and two domain
diagonal propagation times have elapsed. All bins and the residual must pass for
three consecutive checks. Empty scattering is handled by the incident-scaled
absolute tolerance. Bins far outside the Gaussian pulse bandwidth are rejected.
The resource cap is checked exactly, including a partial final graph batch.
Hitting it produces `ConvergenceError`, not a successful reference.

The default policy is a practical settling test, not a rigorous bound on the
infinite-time tail. For narrow resonances, increase the check spacing,
consecutive-check count and safety cap, and demonstrate invariance under stronger
policies. A zero excitation or nonfinite state cannot qualify.

Checkpoint records are immutable. A release/acquire device counter publishes
completed records. A separate low-priority, nonblocking stream reads the latest
record and asynchronously copies it to pinned host memory. That stream waits only
for initialization; the FDTD stream never waits for telemetry or a callback.
Host polling is throttled to 100 ms; reports may be coalesced, and short runs may
show only their final report. Shared GPU resources still impose normal scheduling
overhead; there is no claim of zero telemetry overhead. A callback exception is
reported after the autonomous GPU run finishes.

Only final amplitude/width arrays and small diagnostics are downloaded normally.
The explicit validation download is performed after GPU NF2FF. The convergence
history columns are `step, time, error_ratio, residual, stable_checks, status,
incident_peak, generation`; status codes are 0 running, 1 converged, 2 cap,
3 nonfinite. `error_ratio` is the worst bin's change divided by its tolerance,
not a direct estimate of scattering error.
