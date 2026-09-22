# Off-grid PEC and experimental cell enlargement

Implemented for the new CPU TMz solver on 22 September 2026. Select
`pec_mode="conformal"`, `"enlarged"`, or `"staircase"` in `simulate`.
Circles and rectangles use `PEC()` as their material. There are no PEC anchors.

## Cut-edge formulation

In z-invariant TMz, a cut magnetic face becomes an open x/y interval. The
Faraday update divides by the actual vacuum interval length and uses zero total
Ez at the actual PEC intersection. In scattered-field variables the boundary
value is Es=-Ei evaluated at that intersection, not at a displaced interior node.
Electric curl updates initially retain their nonuniform Yee dual metrics.

Circle/rectangle intersections are computed analytically. This avoids uncertainty
from a sampling-based estimate when the open fraction is very small. Coordinates
within 64 machine epsilons times domain length are recognized as coincident; there
is no finite-fraction clipping. Grid nodes and resolved physical PEC intersections
are unchanged. Geometry requiring two disconnected open intervals on one edge is
rejected pending additional magnetic degrees of freedom.

The unaggregated electric wave operator is similar to a positive symmetric matrix
under dual-area weights. A conservative Gershgorin bound determines dt. Very small
open intervals can therefore cause a severe explicit time-step restriction.
This is consistent with the small-cell problem discussed by
[Nieter et al., 2009](https://doi.org/10.1016/j.jcp.2009.07.025).

## Implemented enlargement

This is an independently derived **local TMz Galerkin aggregation**. It is not a
claimed reproduction of a full 3D face-borrowing algorithm. It shares the purpose
of conformal cell enlargement: combine the dynamics of a tiny cut region with a
neighbor while retaining its physical boundary. Published alternatives include
[Zagorodnov, Schuhmann and Weiland, 2007](https://doi.org/10.1016/j.jcp.2007.02.002) and the methods cited in
[Benkler et al.'s conformal PEC report](https://speag.swiss/assets/downloads/publications/ursi2006_benkler_fdtd.pdf).
The equations below define this prototype; it is not the published USC/SC scheme.

For an exterior electric node A only length l from a PEC boundary, choose a
neighbor B one full interval h farther into vacuum. With total E=0 at the boundary,
constrain

    E_total(A) = w * E_total(B),   w = l/(l+h).

This extends the local field variation over the combined length l+h. Constraints
are considered for open fractions below 0.5. The neighbor must be farther from PEC;
acyclic chains are resolved to one master per constrained node. Narrow gaps that
cannot support this relation are not silently closed.

Let E_total=P*q_total. Project **both** electric mass and curl coupling:

    M_reduced = P.T * M * P
    K_reduced = P.T * K * P

Each row of P contains at most one nonzero entry, so M_reduced remains diagonal.
Time stepping uses local weighted transfers to shared master nodes; the sparse
stiffness matrix is built once for the stability bound. It is not multiplied each
time step. Symmetry and positive energy of the undamped, unforced system are
preserved under this projection. This does not independently prove CPML stability.

Scattered-field constraints require an affine incident correction:

    Es = P*q_scattered + g(t),   g(t) = P*Ei_at_masters(t) - Ei_at_nodes(t).

The reduced electric update includes `-P.T*M*(g_new-g_old)`. Omitting it would
change the incident excitation. The magnetic update still uses the true cut length
and actual scattered boundary values. There is no artificial PEC displacement.

The reduced operator gets a new Gershgorin CFL check. The chosen dt is the smaller
of its bound and the background-grid bound, with the same safety factor. Full grid
dt is a measured result for the pilot, not an unconditional promise for all scenes.
No minimum open fraction is substituted into the geometry to obtain that result.

## Tests and interpretation

- Independently assemble the full electric operator on a small nonuniform grid.
  Verify the implementation equals P.T*K*P and P.T*M*P, and bound its actual
  eigenvalues. This tests the coupling and mass together.
- Check total-field constraints while the incident pulse is active.
- Compare circle scattering against the analytic PEC-cylinder series at multiple
  mesh resolutions, incidence angles, subcell positions, and three frequencies.
- Compare plain conformal versus enlarged runs at identical geometry/grid.
- Test rectangle cut fractions down to 1e-6 with longer post-pulse simulation.
- Reject unresolved PEC, unsupported split intervals, and mixed dielectric/PEC
  input instead of generating misleading results.

Run `scripts/pilot_conformal_pec.py`; see [progress](../PROGRESS.md) for the measured
table and `runs/conformal_pec_pilot/` for raw spectra, report, and plots. Reported
runtime includes CPU setup and streaming observations and is not a GPU benchmark.

Enlargement changes the approximation near the boundary and removes some local
degrees of freedom. Accuracy must be compared at equal budgets and actual runtime,
not inferred from restored dt alone. Thin screens, subcell narrow gaps, mixed
PEC/dielectric coupling, strong-grading PEC stress cases, cavities, and resonances
remain future qualification. The default stays plain conformal for now; enlarged
mode is explicit while this evidence is collected.
