# cython: language_level=3
"""Validated boundary to a native CUDA run; releases the GIL for the whole run."""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint64_t

cdef extern from "fdtd_api.h":
    cdef struct RunStats:
        double milliseconds
        uint64_t h2d_bytes
        uint64_t d2h_bytes
        uint64_t stepping_transfers
    int fdtd_device_count() nogil
    int fdtd_run(int, int, int, int, int, int,
        const void*, const void*, const void*, const void*, const void*,
        const unsigned char*, const int*, const void*, const int*,
        void*, void*, void*, void*, RunStats*, char*, int) nogil

def device_count():
    return fdtd_device_count()

def run(c, initial, source_indices, waveforms, receiver_indices):
    for indices in (source_indices, receiver_indices):
        raw = np.asarray(indices)
        if (raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu")
                or np.any(raw < 0) or np.any(raw > 2147483647)):
            raise ValueError("Runtime indices must be nonnegative int32-compatible integers")
    cdef cnp.ndarray ez = np.array(initial[0], copy=True, order="C")
    cdef cnp.ndarray hx = np.array(initial[1], copy=True, order="C")
    cdef cnp.ndarray hy = np.array(initial[2], copy=True, order="C")
    cdef cnp.ndarray ca = np.ascontiguousarray(c.ca)
    cdef cnp.ndarray cbx = np.ascontiguousarray(c.cbx)
    cdef cnp.ndarray cby = np.ascontiguousarray(c.cby)
    cdef cnp.ndarray chx = np.ascontiguousarray(c.chx)
    cdef cnp.ndarray chy = np.ascontiguousarray(c.chy)
    cdef cnp.ndarray pec = np.ascontiguousarray(c.pec, dtype=np.uint8)
    cdef cnp.ndarray sources = np.ascontiguousarray(source_indices, dtype=np.int32)
    cdef cnp.ndarray waves = np.ascontiguousarray(waveforms)
    cdef cnp.ndarray receivers = np.ascontiguousarray(receiver_indices, dtype=np.int32)
    if ez.ndim != 2 or waves.ndim != 2 or sources.ndim != 1 or receivers.ndim != 1:
        raise ValueError("Invalid runtime array dimensions")
    if max(ez.size, waves.shape[0], sources.size, receivers.size) > 2147483647:
        raise ValueError("Runtime sizes exceed int32 capacity")
    cdef int nx = <int>ez.shape[0]-1, ny = <int>ez.shape[1]-1
    cdef int nt = <int>waves.shape[0], ns = <int>sources.size, nr = <int>receivers.size
    if nx<1 or ny<1 or nt<1 or waves.shape[1]!=ns or ez.size>2147483647:
        raise ValueError("Invalid runtime sizes")
    cdef int precision = 32 if ez.dtype == np.float32 else 64
    if ez.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Expected float32 or float64")
    for a, shape in [(hx,(nx+1,ny)),(hy,(nx,ny+1)),(ca,(nx+1,ny+1)),
                     (cbx,(nx+1,ny+1)),(cby,(nx+1,ny+1)),(chx,(nx+1,ny)),
                     (chy,(nx,ny+1)),(waves,(nt,ns))]:
        if a.shape != shape or a.dtype != ez.dtype or not np.isfinite(a).all():
            raise ValueError("Invalid runtime shapes, dtypes, or nonfinite data")
    if (<object>pec).shape != (<object>ez).shape or not np.isfinite(ez).all():
        raise ValueError("Invalid PEC mask or initial field")
    if (np.any(sources<0) or np.any(sources>=ez.size) or np.any(receivers<0)
            or np.any(receivers>=ez.size) or len(np.unique(sources))!=ns):
        raise ValueError("Runtime source/receiver indices out of range or duplicated sources")
    if np.any(pec.ravel()[sources]):
        raise ValueError("Sources cannot occupy PEC")
    ez[pec.astype(bool)] = 0
    cdef cnp.ndarray history = np.empty((nt,nr), dtype=ez.dtype)
    cdef RunStats stats
    cdef char error[1024]
    cdef int status
    with nogil:
        status = fdtd_run(precision,nx,ny,nt,ns,nr,
            ca.data,cbx.data,cby.data,chx.data,chy.data,<unsigned char*>pec.data,
            <int*>sources.data,waves.data,<int*>receivers.data,
            ez.data,hx.data,hy.data,history.data,&stats,error,1024)
    if status:
        raise RuntimeError((<bytes>error).decode("utf-8", "replace"))
    return ez,hx,hy,history,dict(gpu_ms=stats.milliseconds,h2d_bytes=stats.h2d_bytes,
        d2h_bytes=stats.d2h_bytes,stepping_transfers=stats.stepping_transfers)
