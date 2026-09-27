# API redesign: implemented POC scope

Implemented on 2026-09-27. The runnable reference is [solver_api.md](solver_api.md).

## Decisions carried into the implementation

1. **One workflow:** `Simulation` → add geometry → `apply_mesh` → `solve` →
   `Result.save` / plotting. Low-level numerical interfaces remain available.
2. **Continuous geometry is authoritative.** Immutable ordered circle, rectangle,
   and polygon records live independently of the Yee grid. Material is PEC or air;
   later regions overwrite earlier interiors. Touching regions of the same material
   have no internal interface. Regularized boundary classification avoids phantom
   PEC sheets between adjacent air cuts. Analytic line intersections drive conformal
   coefficients; raster images are derived CNN inputs, never replacement geometry.
3. **SI units** for inputs and archives. Wavelength-normalized display is optional.
   CNN features use normalized domain coordinates and electrical-size conditioning.
4. **Mesh strategies** are uniform, deterministic, supplied density, a supplied
   coordinate mesh, or CNN inference from a validated ONNX checkpoint bundle.
   Generated meshes share exact budgets, fixed collars, layout anchors, and grading
   constraints. There are no trained weights in the POC.
5. **One built-in source:** a Gaussian-modulated cosine designed from `fmin/fmax`,
   with approximately −6 dB band edges and envelope cutoff at ±5τ. Default analysis
   is 21 bins; explicit frequency vectors are supported. The introductory examples
   use 0.9–1.1 GHz. Custom waveforms and controlled-rolloff filters are deferred.
6. **Conformal plus enlarged-cell FDTD only**, using the native CUDA backend.
   Equivalent-current DFT, convergence, and NF2FF remain on the GPU. CPU callbacks
   receive asynchronous telemetry; device execution does not wait for them.
7. **Per-bin history from the same run.** Compact current and incident convergence
   ratios and incident strength are stored on device and downloaded after NF2FF.
   This is diagnostic history, not equivalent-current data. Default solves do not
   download time-domain fields or current arrays.
8. **HDF5** archives hold geometry, configuration, mesh, complex far field, 2D
   scattering width, convergence history, and diagnostics. Result archives load
   and plot without CUDA. Simulation archives can be prepared again on the CPU.

## Deliberately outside this implementation

- Training a CNN or claiming learned-mesh accuracy.
- Custom waveform callbacks, configurable stopband filters, dielectric media,
  arbitrary incidence, 3D RCS, or a CPU production solver.
- Automatically changing the exact geometry to accommodate an unsupported cut cell.

The next training milestone should compare the same continuous geometry across
mesh strategies against separately qualified spatially refined references. Temporal
DFT convergence alone does not establish spatial or PML accuracy.
