# cython: language_level=3
"""Validated native CUDA boundary. Python never advances timesteps or decides stopping."""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint64_t

cdef extern from "runtime_types.h":
    cdef struct RunConfig:
        int nx, ny, a, b, c, d, source, origin, nf, nr, nd
        int max_steps, check_interval, min_steps, stable_checks, auto_stop, debug
        double dt, f0, pulse_width, pulse_delay, source_scale
        double rtol, atol, field_tol, origin_x, origin_y
    cdef struct HostInputs:
        const void *ex, *ey, *ih, *hxq, *hyq, *hxc, *hyc, *profiles
        const int *hxp, *hyp, *monitors
        const unsigned char* pec
        const double *points, *normals, *ds, *weights, *frequencies, *angles
        const void *initial_e, *initial_hx, *initial_hy
    cdef struct HostOutputs:
        double *amplitude, *width, *history
        void *ez, *hx, *hy
        double *currents, *incident
    cdef struct Progress:
        int generation, step, status, stable
        double time, error, residual, incident_peak
    cdef struct RunStats:
        double gpu_ms
        uint64_t h2d_bytes, result_bytes, debug_bytes, telemetry_bytes, device_bytes
        int steps, status, checks, reports, stable
        double error, residual
    ctypedef void (*ProgressCallback)(const Progress*, void*) noexcept
    int fdtd_device_count() nogil
    int fdtd_run(int, const RunConfig*, const HostInputs*, HostOutputs*, RunStats*,
                 ProgressCallback, void*, char*, int) nogil


def device_count():
    return fdtd_device_count()


cdef void notify(const Progress* p, void* context) noexcept with gil:
    box = <object>context
    if box[1] is not None:
        return
    try:
        box[0](dict(generation=p.generation, step=p.step, status=p.status,
                    stable_checks=p.stable, simulated_time=p.time,
                    dft_error_ratio=p.error, residual=p.residual))
    except BaseException as exc:
        # The GPU graph is autonomous even if a presentation callback fails.
        box[1] = exc


cdef void* pointer(object array):
    return (<cnp.ndarray>array).data


def run(c, shape, box, source, contour, options, *, initial=None, progress=None):
    cdef RunConfig cfg
    cdef HostInputs inputs
    cdef HostOutputs outputs
    cdef RunStats stats
    cdef ProgressCallback callback = NULL
    cdef int status, precision
    cdef char error[2048]
    cfg.nx, cfg.ny = shape
    cfg.a, cfg.b, cfg.c, cfg.d = box
    cfg.source, cfg.origin = source, options["origin_index"]
    cfg.max_steps, cfg.check_interval = options["max_steps"], options["check_interval"]
    cfg.min_steps, cfg.stable_checks = options["min_steps"], options["stable_checks"]
    cfg.auto_stop, cfg.debug = options.get("auto_stop", True), options.get("debug", False)
    cfg.dt, cfg.f0 = c.dt, options["frequency"]
    cfg.pulse_width, cfg.pulse_delay = options["pulse_width"], options["pulse_delay"]
    cfg.source_scale = options["source_scale"]
    cfg.rtol, cfg.atol, cfg.field_tol = options["rtol"], options["atol"], options["field_tol"]
    cfg.origin_x, cfg.origin_y = options["origin"]
    freqs = np.ascontiguousarray(options["frequencies"], dtype=np.float64)
    angles = np.ascontiguousarray(options["angles"], dtype=np.float64)
    if freqs.ndim != 1 or angles.ndim != 1 or not len(freqs) or not len(angles):
        raise ValueError("Frequency and angle arrays must be nonempty vectors")
    if (not np.isfinite(freqs).all() or np.any(freqs <= 0) or np.any(freqs*c.dt >= .5)
            or not np.isfinite(angles).all()):
        raise ValueError("Invalid frequency/angle values")
    cfg.nf, cfg.nd, cfg.nr = len(freqs), len(angles), len(contour.indices)
    nx, ny, nr, nf, nd = cfg.nx, cfg.ny, cfg.nr, cfg.nf, cfg.nd
    if (min(nx, ny) < 3 or min(cfg.max_steps, cfg.check_interval, cfg.stable_checks, nr) < 1
            or cfg.max_steps > 10_000_000 or cfg.min_steps < 0
            or max((nx+1)*(ny+1), 4*nf*nr, 2*nf*nd) > 2147483647
            or not 0 < cfg.source < nx or not 0 < cfg.origin < nx
            or not 0 < cfg.a < cfg.b < nx or not 0 < cfg.c < cfg.d < ny):
        raise ValueError("Invalid native dimensions, resource cap or indices")
    if (not np.isfinite([cfg.dt,cfg.f0,cfg.pulse_width,cfg.pulse_delay,cfg.source_scale,
                         cfg.rtol,cfg.atol,cfg.field_tol,cfg.origin_x,cfg.origin_y]).all()
            or min(cfg.dt,cfg.f0,cfg.pulse_width,cfg.rtol,cfg.atol,cfg.field_tol) <= 0):
        raise ValueError("Invalid source or convergence parameters")
    dtype = np.asarray(c.ex).dtype
    if dtype not in (np.dtype("float32"),np.dtype("float64")):
        raise ValueError("Native fields require float32 or float64")
    precision = 32 if dtype == np.float32 else 64
    raw = [c.ex,c.ey,c.ih,c.hx.diagonal,c.hy.diagonal,c.hx.coupling,c.hy.coupling,c.cpml]
    shapes = [(nx+1,),(ny+1,),(nx,),(nx+1,ny),(nx,ny+1),(nx+1,ny),(nx,ny+1),(2*nx+2*ny+2,3)]
    arrays = []
    for a, expected in zip(raw,shapes):
        if np.shape(a) != expected or not np.isfinite(a).all():
            raise ValueError("Malformed coefficient buffer")
        arrays.append(np.ascontiguousarray(a,dtype=dtype))
    peers = []
    for p, expected in zip([c.hx.peer,c.hy.peer],[(nx+1,ny),(nx,ny+1)]):
        p = np.asarray(p)
        if p.shape != expected or p.dtype.kind not in "iu" or np.any(p < -1) or np.any(p >= p.size):
            raise ValueError("Invalid enlarged-cell partner")
        peers.append(np.ascontiguousarray(p,dtype=np.int32))
    mask = np.ascontiguousarray(c.pec,dtype=np.uint8)
    if mask.shape != (nx+1,ny+1) or not mask[[0,-1],:].all() or not mask[:,[0,-1]].all():
        raise ValueError("Invalid PEC/outer-boundary mask")
    monitors = np.asarray(contour.indices)
    if (monitors.shape != (nr,3) or monitors.dtype.kind not in "iu" or np.any(monitors < 0)
            or np.any(monitors[:,0] >= mask.size)
            or np.any(monitors[:,1:] >= (nx+1)*ny+nx*(ny+1))):
        raise ValueError("Invalid monitor index buffer")
    monitors = np.ascontiguousarray(monitors,dtype=np.int32)
    geom = []
    for a, expected in zip([contour.points,contour.normals,contour.ds,contour.weights],
                           [(nr,2),(nr,2),(nr,),(nr,2)]):
        if np.shape(a) != expected or not np.isfinite(a).all():
            raise ValueError("Invalid monitor geometry")
        geom.append(np.ascontiguousarray(a,dtype=np.float64))
    inputs.ex, inputs.ey, inputs.ih = pointer(arrays[0]),pointer(arrays[1]),pointer(arrays[2])
    inputs.hxq,inputs.hyq = pointer(arrays[3]),pointer(arrays[4])
    inputs.hxc,inputs.hyc,inputs.profiles = pointer(arrays[5]),pointer(arrays[6]),pointer(arrays[7])
    inputs.hxp,inputs.hyp = <int*>pointer(peers[0]),<int*>pointer(peers[1])
    inputs.pec,inputs.monitors = <unsigned char*>pointer(mask),<int*>pointer(monitors)
    inputs.points,inputs.normals = <double*>pointer(geom[0]),<double*>pointer(geom[1])
    inputs.ds,inputs.weights = <double*>pointer(geom[2]),<double*>pointer(geom[3])
    inputs.frequencies,inputs.angles = <double*>pointer(freqs),<double*>pointer(angles)
    inputs.initial_e,inputs.initial_hx,inputs.initial_hy = NULL,NULL,NULL
    initial_buffers = []
    if initial is not None:
        if len(initial) != 3:
            raise ValueError("Initial fields must be Ez, Hx, Hy")
        for a, expected in zip(initial,[(nx+1,ny+1),(nx+1,ny),(nx,ny+1)]):
            if np.shape(a) != expected or not np.isfinite(a).all():
                raise ValueError("Invalid initial field")
            initial_buffers.append(np.array(a,dtype=dtype,order="C",copy=True))
        initial_buffers[0][mask.astype(bool)] = 0
        inputs.initial_e = pointer(initial_buffers[0])
        inputs.initial_hx = pointer(initial_buffers[1])
        inputs.initial_hy = pointer(initial_buffers[2])
    amplitude = np.empty((nf,nd),dtype=np.complex128)
    width = np.empty((nf,nd),dtype=np.float64)
    history = np.empty((cfg.max_steps//cfg.check_interval+2,8),dtype=np.float64)
    outputs.amplitude,outputs.width,outputs.history = <double*>pointer(amplitude),<double*>pointer(width),<double*>pointer(history)
    outputs.ez = NULL
    outputs.hx = NULL
    outputs.hy = NULL
    outputs.currents = NULL
    outputs.incident = NULL
    debug = {}
    if cfg.debug:
        debug = dict(Ez=np.empty((nx+1,ny+1),dtype=dtype),Hx=np.empty((nx+1,ny),dtype=dtype),
                     Hy=np.empty((nx,ny+1),dtype=dtype),currents=np.empty((nf,nr,2),dtype=np.complex128),
                     incident=np.empty((nf,nx+1),dtype=np.complex128))
        outputs.ez,outputs.hx,outputs.hy = pointer(debug["Ez"]),pointer(debug["Hx"]),pointer(debug["Hy"])
        outputs.currents,outputs.incident = <double*>pointer(debug["currents"]),<double*>pointer(debug["incident"])
    context = [progress,None]
    if progress is not None:
        if not callable(progress):
            raise ValueError("Progress must be callable")
        callback = notify
    with nogil:
        status = fdtd_run(precision,&cfg,&inputs,&outputs,&stats,callback,<void*>context,error,2048)
    if status:
        raise RuntimeError((<bytes>error).decode("utf-8","replace"))
    if context[1] is not None:
        raise context[1]
    diagnostics = dict(backend="native-cuda",controller="device-conditional-graph",
                       gpu_ms=stats.gpu_ms,h2d_bytes=stats.h2d_bytes,result_d2h_bytes=stats.result_bytes,
                       debug_d2h_bytes=stats.debug_bytes,telemetry_d2h_bytes=stats.telemetry_bytes,
                       device_bytes=stats.device_bytes,stepping_field_transfers=0,
                       Nt=stats.steps,status=["running","converged","max_steps","nonfinite"][stats.status],
                       checks=stats.checks,stable_checks=stats.stable,dft_error_ratio=stats.error,
                       residual=stats.residual)
    return amplitude,width,history[:stats.reports].copy(),diagnostics,debug
