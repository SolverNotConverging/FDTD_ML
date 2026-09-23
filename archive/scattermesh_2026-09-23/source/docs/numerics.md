# Numerical conventions

## Mesh and update equations

Coordinates begin at zero and use metres. There are Nx+1 x nodes and Ny+1 y nodes.
Ez is stored on nodes, Hx on y-edge midpoints, Hy on x-edge midpoints. H updates
use primal edge widths; E updates use neighboring midpoint-to-midpoint distances.
The selected time step is

    dt = safety / (c0 * sqrt(min(dx)^(-2) + min(dy)^(-2)))

with 0 < safety < 1. This conservative vacuum bound applies to the implemented
epsilon_r >= 1, mu_r = 1 media. PEC modes additionally impose their cut-edge or
projected-operator bound. No coordinate or source anchor insertion occurs.
`Grid.max_ratio` validates a selected grading cap; it does not smooth a mesh.

Ez material coefficients are arithmetic filling-fraction averages over dual-cell
areas, estimated by configurable midpoint sampling in each axis. Each sample is
assigned the last containing object's material. PEC is handled by separate
[boundary and enlargement operators](conformal_pec.md), never by averaging.

Let `q = sigma*dt/(2*epsilon)` and `b = (1-epsilon0/epsilon)/(1+q)`.
After the scattered H half-step, the interior E update is

    Es_new = (1-q)/(1+q) * Es_old + dt/(epsilon*(1+q)) * curl(Hs)
             - b * (Ei_new - Ei_old) - q/(1+q) * (Ei_new + Ei_old)

The incident plane wave is analytic and is evaluated at material nodes only in
dielectric scenes. PEC uses scattered boundary values equal to minus the incident
field at the actual PEC intersection.
The Gaussian envelope must be negligible at all nodes at t=0. Plane-wave
propagation is defined by angle counterclockwise from +x; Ez polarization is
perpendicular to every in-plane incidence direction.

CPML stretches scattered-field derivatives near the domain perimeter. Its
coefficients are sampled at actual E/H positions with a physical layer thickness.
The outermost E nodes remain zero. A nominal profile reflection parameter of
1e-8 is not a measured reflection guarantee, especially on coarse/graded PML cells.

## DFT and near-to-far conventions

The Fourier transform uses integral f(t) exp(+i omega t) dt; phasors have time
dependence exp(-i omega t). Electric and magnetic contributions are accumulated
at their actual times (n+1)dt and (n+1/2)dt. Interpolation uses the nonuniform
neighbor distances, and contour quadrature uses physical segment lengths.

The closed rectangle selects existing mesh nodes. Its interpolation stencil must
be outside PML, and every object must be enclosed with vacuum clearance. No mesh
lines are inserted to place the contour. The exterior must be homogeneous vacuum.

For counterclockwise contour tangent t=(-ny,nx), define Ht=H dot t. Then

    Es_far(r, phi) ~ A(phi) * exp(i*k*r) / sqrt(r)
    A(phi) = exp(i*pi/4)/sqrt(8*pi*k)
             * integral i*(omega*mu0*Ht - k*(n dot rhat)*Es)
                        * exp(-i*k*(rhat dot r')) dl

Source normalization is the DFT of the incident pulse at its phase origin.
`SurfaceDFT.normalized_far_field` returns complex A/Einc in sqrt(m); this is the
primary training/reference target. It uses the global far-field coordinate origin
(0,0). Both origins and the Fourier convention are recorded in solver diagnostics.
The angular **2D scattering width** is

    width(phi) = 2*pi*abs(A(phi)/Ei_origin)^2    [metres]

The differential cross section per radian is width/(2*pi); total scattering cross
section per unit length is its integral over angle. For uniformly sampled angles
covering 2*pi, this integral is the mean width. Complex amplitudes depend on phase
origin/target position even when translated-target widths agree.

For a target translated by dr at fixed illumination phase origin, its complex
normalized amplitude gains exp(i*k*(d_incident-d_observation) dot dr). The analytic
cylinder benchmark includes this factor. Source normalization does not remove
physical scattering phase, and predictions are never rotated to match references.

Store real and imaginary components together (the pilots use native complex128
NPZ arrays). The complex L2 metric measures amplitude and phase simultaneously.
Weighted phase RMS is an additional diagnostic over sufficiently strong angular
samples; no phase is reported for a zero field. Per-frequency normalization uses
an explicit amplitude floor, which must be calibrated before training.
Candidate ranking combines this complex error with a floored logarithmic
scattering-width error; see the [joint loss and curriculum](curriculum.md).

Bins below a floor relative to the incident time-domain L1 norm or strongest
requested bin are rejected. Thus even a request containing only out-of-band
frequencies cannot silently normalize numerical noise.

## What diagnostics do and do not mean

The solver returns dt, Nt, grid widths/grading, Nx*Ny*Nt cell updates, elapsed CPU
time, peak scattered Ez, maximum scattered Ez during the last 10% relative to its
global peak, source residual at the phase origin, and observation settings.
Field-tail diagnostics are not energy-convergence proofs or dataset acceptance.
Global field-energy/DFT settling windows and automatic reference escalation remain
to be implemented. A finite low tail does not establish spatial convergence.

An empty domain is identically zero because the source is contrast driven. A
nonempty analytic cylinder and an independent outgoing Hankel wave test source,
propagation, normalization, phase, and NF2FF more meaningfully. The current finite
test set does not qualify epsilon_r=30 resonances, arbitrary PEC topology, all
angles, or all grading. The PEC pilot qualifies only its stated finite examples.
