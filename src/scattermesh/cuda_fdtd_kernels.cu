#include "cuda_fdtd_kernels.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cg = cooperative_groups;

struct DeviceBuffers {
    void *inputs[FDTD_INPUT_COUNT];
    void *outputs[FDTD_OUTPUT_COUNT];
};

struct Config {
    int nx, ny, nt, nf, npoints, nmasters, nslaves, i0, i1, j0, j1, tail_start;
    double dt, mu0, source_frequency, source_delay, source_width;
};

__device__ __forceinline__ const double *real_input(const DeviceBuffers &b, int index) {
    return static_cast<const double *>(b.inputs[index]);
}

__device__ __forceinline__ const int *int_input(const DeviceBuffers &b, int index) {
    return static_cast<const int *>(b.inputs[index]);
}

__device__ __forceinline__ const unsigned char *mask_input(const DeviceBuffers &b, int index) {
    return static_cast<const unsigned char *>(b.inputs[index]);
}

__device__ __forceinline__ double pulse(double time, const Config &c) {
    const double shifted = (time - c.source_delay) / c.source_width;
    return exp(-shifted * shifted) * cos(6.2831853071795864769 * c.source_frequency * (time - c.source_delay));
}

__device__ __forceinline__ double edge_gradient(
    const DeviceBuffers &b, const Config &c, int edge, int low, int high,
    int inverse_index, int exterior_index, int delay_index, int sign_index, double time
) {
    const double inverse = real_input(b, inverse_index)[edge];
    if (inverse == 0.0) return 0.0;
    const int exterior = int_input(b, exterior_index)[edge];
    const double *ez = static_cast<const double *>(b.outputs[OUT_EZ]);
    if (exterior >= 0) {
        return real_input(b, sign_index)[edge] *
               (ez[exterior] + pulse(time - real_input(b, delay_index)[edge], c)) * inverse;
    }
    return (ez[high] - ez[low]) * inverse;
}

__device__ __forceinline__ void contour_sample(
    const DeviceBuffers &b, const Config &c, int point, double *electric, double *magnetic
) {
    const int vertical = c.j1 - c.j0;
    const int horizontal = c.i1 - c.i0;
    const double *ez = static_cast<const double *>(b.outputs[OUT_EZ]);
    const double *hx = static_cast<const double *>(b.outputs[OUT_HX]);
    const double *hy = static_cast<const double *>(b.outputs[OUT_HY]);
    const double *x = real_input(b, IN_X);
    const double *y = real_input(b, IN_Y);
    if (point < 2 * vertical) {
        const bool right_side = point >= vertical;
        const int i = right_side ? c.i1 : c.i0;
        const int j = c.j0 + point % vertical;
        *electric = 0.5 * (ez[i * c.ny + j] + ez[i * c.ny + j + 1]);
        const double left = x[i] - x[i - 1];
        const double right = x[i + 1] - x[i];
        const double h0 = (right * hy[(i - 1) * c.ny + j] + left * hy[i * c.ny + j]) / (left + right);
        const double h1 = (right * hy[(i - 1) * c.ny + j + 1] + left * hy[i * c.ny + j + 1]) / (left + right);
        *magnetic = (right_side ? 1.0 : -1.0) * 0.5 * (h0 + h1);
    } else {
        const int offset = point - 2 * vertical;
        const bool top_side = offset >= horizontal;
        const int i = c.i0 + offset % horizontal;
        const int j = top_side ? c.j1 : c.j0;
        *electric = 0.5 * (ez[i * c.ny + j] + ez[(i + 1) * c.ny + j]);
        const double left = y[j] - y[j - 1];
        const double right = y[j + 1] - y[j];
        const int hy_stride = c.ny - 1;
        const double h0 = (right * hx[i * hy_stride + j - 1] + left * hx[i * hy_stride + j]) / (left + right);
        const double h1 = (right * hx[(i + 1) * hy_stride + j - 1] + left * hx[(i + 1) * hy_stride + j]) / (left + right);
        *magnetic = (top_side ? -1.0 : 1.0) * 0.5 * (h0 + h1);
    }
}

__global__ void persistent_fdtd(DeviceBuffers b, Config c) {
    const cg::grid_group whole_grid = cg::this_grid();
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int nodes = c.nx * c.ny;
    const int hx_count = c.nx * (c.ny - 1);
    const int hy_count = (c.nx - 1) * c.ny;
    const int contour_count = 2 * (c.i1 - c.i0 + c.j1 - c.j0);
    double *ez = static_cast<double *>(b.outputs[OUT_EZ]);
    double *hx = static_cast<double *>(b.outputs[OUT_HX]);
    double *hy = static_cast<double *>(b.outputs[OUT_HY]);
    double *peak = static_cast<double *>(b.outputs[OUT_PEAK_CELLS]);
    double *tail = static_cast<double *>(b.outputs[OUT_TAIL_CELLS]);
    int *status = static_cast<int *>(b.outputs[OUT_STATUS]);
    double *phx = nullptr;
    double *phy = nullptr;
    double *pex = nullptr;
    double *pey = nullptr;
    // CPML state follows the output buffers in one contiguous workspace.
    double *workspace = reinterpret_cast<double *>(
        static_cast<char *>(b.outputs[OUT_STATUS]) + 8
    );
    phx = workspace;
    phy = phx + hx_count;
    pex = phy + hy_count;
    pey = pex + nodes;
    double *delta = pey + nodes;
    double *numerator = delta + nodes;

    for (int k = thread; k < nodes; k += stride) {
        if (mask_input(b, IN_INSIDE)[k]) {
            ez[k] = -pulse(-real_input(b, IN_DELAY)[k], c);
        }
    }
    whole_grid.sync();
    if (c.nslaves) {
        for (int s = thread; s < c.nslaves; s += stride) {
            const int slave = int_input(b, IN_SLAVES)[s];
            const int root = int_input(b, IN_ROOTS)[s];
            const double weight = real_input(b, IN_SLAVE_WEIGHTS)[s];
            const double offset = weight * pulse(-real_input(b, IN_ROOT_DELAY)[s], c) -
                                  pulse(-real_input(b, IN_SLAVE_DELAY)[s], c);
            ez[slave] = weight * ez[root] + offset;
        }
        whole_grid.sync();
    }

    for (int n = 0; n < c.nt; ++n) {
        const double magnetic_time = n * c.dt;
        const double electric_time = (n + 1) * c.dt;
        for (int k = thread; k < hx_count; k += stride) {
            const int i = k / (c.ny - 1);
            const int j = k % (c.ny - 1);
            const double gy = edge_gradient(
                b, c, k, i * c.ny + j, i * c.ny + j + 1,
                IN_INV_Y, IN_BOUNDARY_EXT_Y, IN_BOUNDARY_DELAY_Y, IN_BOUNDARY_SIGN_Y,
                magnetic_time
            );
            phx[k] = real_input(b, IN_HPY_B)[j] * phx[k] + real_input(b, IN_HPY_A)[j] * gy;
            hx[k] -= c.dt / c.mu0 * (real_input(b, IN_HPY_K)[j] * gy + phx[k]);
        }
        for (int k = thread; k < hy_count; k += stride) {
            const int i = k / c.ny;
            const int j = k % c.ny;
            const double gx = edge_gradient(
                b, c, k, i * c.ny + j, (i + 1) * c.ny + j,
                IN_INV_X, IN_BOUNDARY_EXT_X, IN_BOUNDARY_DELAY_X, IN_BOUNDARY_SIGN_X,
                magnetic_time
            );
            phy[k] = real_input(b, IN_HPX_B)[i] * phy[k] + real_input(b, IN_HPX_A)[i] * gx;
            hy[k] += c.dt / c.mu0 * (real_input(b, IN_HPX_K)[i] * gx + phy[k]);
        }
        whole_grid.sync();

        for (int k = thread; k < nodes; k += stride) {
            const int i = k / c.ny;
            const int j = k % c.ny;
            if (i > 0 && i < c.nx - 1 && j > 0 && j < c.ny - 1) {
                const double gx = (hy[i * c.ny + j] - hy[(i - 1) * c.ny + j]) * real_input(b, IN_DUAL_X)[i - 1];
                const double gy = (hx[i * (c.ny - 1) + j] - hx[i * (c.ny - 1) + j - 1]) * real_input(b, IN_DUAL_Y)[j - 1];
                pex[k] = real_input(b, IN_EX_B)[i] * pex[k] + real_input(b, IN_EX_A)[i] * gx;
                pey[k] = real_input(b, IN_EY_B)[j] * pey[k] + real_input(b, IN_EY_A)[j] * gy;
                const double curl = real_input(b, IN_EX_K)[i] * gx + pex[k] -
                                    real_input(b, IN_EY_K)[j] * gy - pey[k];
                if (c.nslaves) {
                    delta[k] = (real_input(b, IN_CA)[k] - 1.0) * ez[k] +
                               real_input(b, IN_CB)[k] * curl;
                } else {
                    ez[k] = real_input(b, IN_CA)[k] * ez[k] + real_input(b, IN_CB)[k] * curl;
                }
            }
        }
        whole_grid.sync();
        if (c.nslaves) {
            for (int m = thread; m < c.nmasters; m += stride) {
                numerator[m] = real_input(b, IN_MASTER_MASS)[m] *
                               delta[int_input(b, IN_MASTERS)[m]];
            }
            whole_grid.sync();
            for (int s = thread; s < c.nslaves; s += stride) {
                const double weight = real_input(b, IN_SLAVE_WEIGHTS)[s];
                const double root_delay = real_input(b, IN_ROOT_DELAY)[s];
                const double slave_delay = real_input(b, IN_SLAVE_DELAY)[s];
                const double g = weight * pulse(electric_time - root_delay, c) -
                                 pulse(electric_time - slave_delay, c);
                const double previous_g = weight * pulse(magnetic_time - root_delay, c) -
                                          pulse(magnetic_time - slave_delay, c);
                const double transferred = real_input(b, IN_SLAVE_MASS_WEIGHT)[s] *
                    (delta[int_input(b, IN_SLAVES)[s]] - (g - previous_g));
                atomicAdd(&numerator[int_input(b, IN_SLAVE_OWNERS)[s]], transferred);
            }
            whole_grid.sync();
            for (int m = thread; m < c.nmasters; m += stride) {
                ez[int_input(b, IN_MASTERS)[m]] +=
                    numerator[m] / real_input(b, IN_AGGREGATE_MASS)[m];
            }
            whole_grid.sync();
            for (int s = thread; s < c.nslaves; s += stride) {
                const double weight = real_input(b, IN_SLAVE_WEIGHTS)[s];
                const double g = weight * pulse(electric_time - real_input(b, IN_ROOT_DELAY)[s], c) -
                                 pulse(electric_time - real_input(b, IN_SLAVE_DELAY)[s], c);
                ez[int_input(b, IN_SLAVES)[s]] = weight *
                    ez[int_input(b, IN_ROOTS)[s]] + g;
            }
            whole_grid.sync();
        }
        for (int k = thread; k < nodes; k += stride) {
            if (mask_input(b, IN_ACTIVE)[k]) {
                const double delay = real_input(b, IN_DELAY)[k];
                const double incident = pulse(electric_time - delay, c);
                const double previous = pulse(magnetic_time - delay, c);
                ez[k] -= real_input(b, IN_CONTRAST)[k] * (incident - previous) +
                         real_input(b, IN_CONDUCTION)[k] * (incident + previous);
            }
            if (mask_input(b, IN_INSIDE)[k]) {
                ez[k] = -pulse(electric_time - real_input(b, IN_DELAY)[k], c);
            }
            const double magnitude = fabs(ez[k]);
            if (!isfinite(magnitude) || magnitude > 1e8) atomicExch(status, 1);
            peak[k] = fmax(peak[k], magnitude);
            if (n >= c.tail_start) tail[k] = fmax(tail[k], magnitude);
        }
        whole_grid.sync();

        double *edft = static_cast<double *>(b.outputs[OUT_ELECTRIC_DFT]);
        double *hdft = static_cast<double *>(b.outputs[OUT_MAGNETIC_DFT]);
        for (int q = thread; q < c.nf * contour_count; q += stride) {
            const int f = q / contour_count;
            const int point = q % contour_count;
            double electric, magnetic;
            contour_sample(b, c, point, &electric, &magnetic);
            const int phase_index = n * c.nf + f;
            edft[2 * q] += c.dt * real_input(b, IN_PHASE_E_REAL)[phase_index] * electric;
            edft[2 * q + 1] += c.dt * real_input(b, IN_PHASE_E_IMAG)[phase_index] * electric;
            hdft[2 * q] += c.dt * real_input(b, IN_PHASE_H_REAL)[phase_index] * magnetic;
            hdft[2 * q + 1] += c.dt * real_input(b, IN_PHASE_H_IMAG)[phase_index] * magnetic;
        }
        double *point_scattered = static_cast<double *>(b.outputs[OUT_POINT_SCATTERED_DFT]);
        double *point_incident = static_cast<double *>(b.outputs[OUT_POINT_INCIDENT_DFT]);
        for (int q = thread; q < c.nf * c.npoints; q += stride) {
            const int f = q / c.npoints;
            const int p = q % c.npoints;
            const int i = int_input(b, IN_POINT_I)[p];
            const int j = int_input(b, IN_POINT_J)[p];
            const double wx = real_input(b, IN_POINT_WX)[p];
            const double wy = real_input(b, IN_POINT_WY)[p];
            const double value = ez[i * c.ny + j] * (1 - wx) * (1 - wy) +
                ez[(i + 1) * c.ny + j] * wx * (1 - wy) +
                ez[i * c.ny + j + 1] * (1 - wx) * wy +
                ez[(i + 1) * c.ny + j + 1] * wx * wy;
            const double incident = pulse(electric_time - real_input(b, IN_POINT_DELAY)[p], c);
            const int phase_index = n * c.nf + f;
            point_scattered[2 * q] += c.dt * real_input(b, IN_PHASE_E_REAL)[phase_index] * value;
            point_scattered[2 * q + 1] += c.dt * real_input(b, IN_PHASE_E_IMAG)[phase_index] * value;
            point_incident[2 * q] += c.dt * real_input(b, IN_PHASE_E_REAL)[phase_index] * incident;
            point_incident[2 * q + 1] += c.dt * real_input(b, IN_PHASE_E_IMAG)[phase_index] * incident;
        }
        double *incident_dft = static_cast<double *>(b.outputs[OUT_INCIDENT_DFT]);
        for (int f = thread; f < c.nf; f += stride) {
            const int phase_index = n * c.nf + f;
            const double incident = real_input(b, IN_INCIDENT_ORIGIN)[n];
            incident_dft[2 * f] += c.dt * real_input(b, IN_PHASE_E_REAL)[phase_index] * incident;
            incident_dft[2 * f + 1] += c.dt * real_input(b, IN_PHASE_E_IMAG)[phase_index] * incident;
        }
        whole_grid.sync();
        if (*status != 0) break;
    }
}

static int fail(cudaError_t code, char *error, int capacity) {
    if (code == cudaSuccess) return 0;
    std::snprintf(error, capacity, "%s", cudaGetErrorString(code));
    return 1;
}

extern "C" int scattermesh_cuda_fdtd_run(const FdtdHost *host, FdtdRunStats *stats,
                                           char *error, int error_capacity) {
    std::memset(stats, 0, sizeof(*stats));
    if (error_capacity > 0) error[0] = '\0';
    cudaError_t result = cudaSetDevice(host->device_index);
    if (fail(result, error, error_capacity)) return 1;
    cudaDeviceProp properties;
    result = cudaGetDeviceProperties(&properties, host->device_index);
    if (fail(result, error, error_capacity)) return 1;
    if (!properties.cooperativeLaunch) {
        std::snprintf(error, error_capacity, "GPU does not support cooperative kernel launch");
        return 1;
    }
    DeviceBuffers device{};
    std::vector<void *> allocations;
    cudaEvent_t start = nullptr, stop = nullptr;
    int failed = 0;
    for (int i = 0; i < FDTD_INPUT_COUNT && !failed; ++i) {
        if (host->input_bytes[i] == 0) continue;
        result = cudaMalloc(&device.inputs[i], host->input_bytes[i]);
        if (fail(result, error, error_capacity)) { failed = 1; break; }
        allocations.push_back(device.inputs[i]);
        stats->allocated_bytes += host->input_bytes[i];
        result = cudaMemcpy(device.inputs[i], host->inputs[i], host->input_bytes[i], cudaMemcpyHostToDevice);
        if (fail(result, error, error_capacity)) { failed = 1; break; }
        stats->host_to_device_bytes += host->input_bytes[i];
    }
    for (int i = 0; i < FDTD_OUTPUT_COUNT && !failed; ++i) {
        size_t count = host->output_bytes[i];
        if (i == OUT_STATUS) {
            count = 8 + (static_cast<size_t>(host->nx) * (host->ny - 1) +
                      static_cast<size_t>(host->nx - 1) * host->ny +
                      3 * static_cast<size_t>(host->nx) * host->ny +
                      static_cast<size_t>(host->nmasters)) * sizeof(double);
        }
        if (count == 0) continue;
        result = cudaMalloc(&device.outputs[i], count);
        if (fail(result, error, error_capacity)) { failed = 1; break; }
        allocations.push_back(device.outputs[i]);
        stats->allocated_bytes += count;
        result = cudaMemset(device.outputs[i], 0, count);
        if (fail(result, error, error_capacity)) { failed = 1; break; }
    }
    Config config{
        host->nx, host->ny, host->nt, host->nf, host->npoints,
        host->nmasters, host->nslaves,
        host->i0, host->i1, host->j0, host->j1, static_cast<int>(0.9 * host->nt),
        host->dt, host->mu0, host->source_frequency, host->source_delay, host->source_width
    };
    if (!failed) {
        int resident_blocks = 0;
        const int threads = 256;
        result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, persistent_fdtd, threads, 0
        );
        if (fail(result, error, error_capacity)) failed = 1;
        const int contour = 2 * (host->i1 - host->i0 + host->j1 - host->j0);
        const int work = std::max({host->nx * host->ny, host->nf * contour,
                                   host->nf * host->npoints});
        const int capacity = resident_blocks * properties.multiProcessorCount;
        const int blocks = std::min((work + threads - 1) / threads, capacity);
        if (!failed && blocks < 1) {
            std::snprintf(error, error_capacity, "No resident blocks for cooperative launch");
            failed = 1;
        }
        if (!failed) {
            stats->cooperative_blocks = blocks;
            stats->threads_per_block = threads;
            result = cudaEventCreate(&start);
            if (fail(result, error, error_capacity)) failed = 1;
        }
        if (!failed) {
            result = cudaEventCreate(&stop);
            if (fail(result, error, error_capacity)) failed = 1;
        }
        if (!failed) {
            result = cudaEventRecord(start);
            if (fail(result, error, error_capacity)) failed = 1;
        }
        if (!failed) {
            void *arguments[] = {&device, &config};
            result = cudaLaunchCooperativeKernel(
                reinterpret_cast<void *>(persistent_fdtd), blocks, threads, arguments
            );
            if (fail(result, error, error_capacity)) failed = 1;
        }
        if (!failed) {
            result = cudaEventRecord(stop);
            if (fail(result, error, error_capacity)) failed = 1;
        }
        if (!failed) {
            result = cudaEventSynchronize(stop);
            if (fail(result, error, error_capacity)) failed = 1;
        }
        if (!failed) {
            float milliseconds = 0.0f;
            result = cudaEventElapsedTime(&milliseconds, start, stop);
            if (fail(result, error, error_capacity)) failed = 1;
            stats->kernel_seconds = milliseconds / 1000.0;
        }
    }
    for (int i = 0; i < FDTD_OUTPUT_COUNT && !failed; ++i) {
        if (host->output_bytes[i] == 0) continue;
        result = cudaMemcpy(host->outputs[i], device.outputs[i], host->output_bytes[i],
                            cudaMemcpyDeviceToHost);
        if (fail(result, error, error_capacity)) { failed = 1; break; }
        stats->device_to_host_bytes += host->output_bytes[i];
    }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    for (void *pointer : allocations) cudaFree(pointer);
    return failed;
}
