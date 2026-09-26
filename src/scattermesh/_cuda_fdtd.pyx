"""Cython binding for the persistent CUDA TMz time loop."""

cimport numpy as cnp
from libc.string cimport memset, strlen


cdef extern from "cuda_fdtd_kernels.h":
    enum:
        FDTD_INPUT_COUNT
        FDTD_OUTPUT_COUNT

    ctypedef struct FdtdHost:
        int nx, ny, nt, nf, npoints, nmasters, nslaves, i0, i1, j0, j1, device_index
        double dt, mu0, source_frequency, source_delay, source_width
        void *inputs[FDTD_INPUT_COUNT]
        size_t input_bytes[FDTD_INPUT_COUNT]
        void *outputs[FDTD_OUTPUT_COUNT]
        size_t output_bytes[FDTD_OUTPUT_COUNT]

    ctypedef struct FdtdRunStats:
        double kernel_seconds
        size_t host_to_device_bytes, device_to_host_bytes, allocated_bytes
        int cooperative_blocks, threads_per_block

    int scattermesh_cuda_fdtd_run(const FdtdHost *host, FdtdRunStats *stats,
                                  char *error, int error_capacity) nogil


def run(dict parameters, list inputs, list outputs):
    """Copy prepared arrays once, execute all FDTD steps on device, copy results once."""
    cdef FdtdHost host
    cdef FdtdRunStats stats
    cdef char error[512]
    cdef int index, result
    cdef cnp.ndarray array
    if len(inputs) != FDTD_INPUT_COUNT or len(outputs) != FDTD_OUTPUT_COUNT:
        raise ValueError("Compiled CUDA buffer schema mismatch")
    memset(&host, 0, sizeof(host))
    memset(&stats, 0, sizeof(stats))
    memset(error, 0, sizeof(error))
    host.nx = parameters["nx"]
    host.ny = parameters["ny"]
    host.nt = parameters["nt"]
    host.nf = parameters["nf"]
    host.npoints = parameters["npoints"]
    host.nmasters = parameters["nmasters"]
    host.nslaves = parameters["nslaves"]
    host.i0 = parameters["i0"]
    host.i1 = parameters["i1"]
    host.j0 = parameters["j0"]
    host.j1 = parameters["j1"]
    host.device_index = parameters["device_index"]
    host.dt = parameters["dt"]
    host.mu0 = parameters["mu0"]
    host.source_frequency = parameters["source_frequency"]
    host.source_delay = parameters["source_delay"]
    host.source_width = parameters["source_width"]
    for index in range(FDTD_INPUT_COUNT):
        array = <cnp.ndarray>inputs[index]
        if not cnp.PyArray_ISCONTIGUOUS(array):
            raise ValueError("CUDA inputs must be contiguous")
        host.inputs[index] = cnp.PyArray_DATA(array)
        host.input_bytes[index] = cnp.PyArray_NBYTES(array)
    for index in range(FDTD_OUTPUT_COUNT):
        array = <cnp.ndarray>outputs[index]
        if not cnp.PyArray_ISCONTIGUOUS(array):
            raise ValueError("CUDA outputs must be contiguous")
        host.outputs[index] = cnp.PyArray_DATA(array)
        host.output_bytes[index] = cnp.PyArray_NBYTES(array)
    with nogil:
        result = scattermesh_cuda_fdtd_run(&host, &stats, error, sizeof(error))
    if result:
        raise RuntimeError(error[:strlen(error)].decode("utf-8"))
    return {
        "kernel_seconds": stats.kernel_seconds,
        "host_to_device_bytes": stats.host_to_device_bytes,
        "device_to_host_bytes": stats.device_to_host_bytes,
        "allocated_bytes": stats.allocated_bytes,
        "cooperative_blocks": stats.cooperative_blocks,
        "threads_per_block": stats.threads_per_block,
        "host_transfers_during_steps": 0,
    }
