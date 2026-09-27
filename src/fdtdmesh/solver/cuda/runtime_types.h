#pragma once
#include <cstdint>
struct RunConfig {
    int nx, ny, a, b, c, d, source, origin, nf, nr, nd;
    int max_steps, check_interval, min_steps, stable_checks, auto_stop, debug;
    double dt, f0, pulse_width, pulse_delay, source_scale, pulse_end;
    double rtol, atol, field_tol, origin_x, origin_y;
};
struct HostInputs {
    const void *ex, *ey, *ih, *hxq, *hyq, *hxc, *hyc, *profiles;
    const int *hxp, *hyp, *monitors;
    const unsigned char* pec;
    const double *points, *normals, *ds, *weights, *frequencies, *angles;
    const void *initial_e, *initial_hx, *initial_hy;
};
struct HostOutputs {
    double *amplitude, *width, *history, *bin_history;
    void *ez, *hx, *hy;
    double *currents, *incident;
};
struct Progress {
    int generation, step, status, stable;
    double time, error, residual, incident_peak;
};
struct RunStats {
    double gpu_ms;
    std::uint64_t h2d_bytes, result_bytes, debug_bytes, telemetry_bytes, device_bytes;
    int steps, status, checks, reports, stable;
    double error, residual;
};
typedef void (*ProgressCallback)(const Progress*, void*);
int fdtd_device_count();
int fdtd_run(int precision, const RunConfig*, const HostInputs*, HostOutputs*,
             RunStats*, ProgressCallback, void*, char*, int);
