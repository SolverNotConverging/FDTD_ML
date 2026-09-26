#ifndef SCATTERMESH_CUDA_FDTD_KERNELS_H
#define SCATTERMESH_CUDA_FDTD_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FDTD_INPUT_COUNT 51
#define FDTD_OUTPUT_COUNT 11

/* Keep the order synchronized with INPUT_NAMES in compiled_cuda.py. */
enum FdtdInput {
    IN_CA, IN_CB, IN_CONTRAST, IN_CONDUCTION, IN_DELAY, IN_ACTIVE, IN_INSIDE,
    IN_INV_X, IN_INV_Y, IN_BOUNDARY_EXT_X, IN_BOUNDARY_EXT_Y,
    IN_BOUNDARY_DELAY_X, IN_BOUNDARY_DELAY_Y, IN_BOUNDARY_SIGN_X, IN_BOUNDARY_SIGN_Y,
    IN_DUAL_X, IN_DUAL_Y,
    IN_EX_K, IN_EX_B, IN_EX_A, IN_EY_K, IN_EY_B, IN_EY_A,
    IN_HPX_K, IN_HPX_B, IN_HPX_A, IN_HPY_K, IN_HPY_B, IN_HPY_A,
    IN_X, IN_Y, IN_PHASE_E_REAL, IN_PHASE_E_IMAG,
    IN_PHASE_H_REAL, IN_PHASE_H_IMAG, IN_INCIDENT_ORIGIN,
    IN_POINT_I, IN_POINT_J, IN_POINT_WX, IN_POINT_WY, IN_POINT_DELAY,
    IN_MASTERS, IN_SLAVES, IN_ROOTS, IN_SLAVE_OWNERS, IN_SLAVE_WEIGHTS,
    IN_MASTER_MASS, IN_SLAVE_MASS_WEIGHT, IN_AGGREGATE_MASS,
    IN_ROOT_DELAY, IN_SLAVE_DELAY
};
enum FdtdOutput {
    OUT_EZ, OUT_HX, OUT_HY, OUT_ELECTRIC_DFT, OUT_MAGNETIC_DFT,
    OUT_INCIDENT_DFT, OUT_POINT_SCATTERED_DFT, OUT_POINT_INCIDENT_DFT,
    OUT_PEAK_CELLS, OUT_TAIL_CELLS, OUT_STATUS
};

typedef struct FdtdHost {
    int nx, ny, nt, nf, npoints, nmasters, nslaves, i0, i1, j0, j1, device_index;
    double dt, mu0, source_frequency, source_delay, source_width;
    void *inputs[FDTD_INPUT_COUNT];
    size_t input_bytes[FDTD_INPUT_COUNT];
    void *outputs[FDTD_OUTPUT_COUNT];
    size_t output_bytes[FDTD_OUTPUT_COUNT];
} FdtdHost;

typedef struct FdtdRunStats {
    double kernel_seconds;
    size_t host_to_device_bytes, device_to_host_bytes, allocated_bytes;
    int cooperative_blocks, threads_per_block;
} FdtdRunStats;

int scattermesh_cuda_fdtd_run(const FdtdHost *host, FdtdRunStats *stats,
                              char *error, int error_capacity);

#ifdef __cplusplus
}
#endif
#endif
