#pragma once
#include <cstdint>
// All arrays are C-contiguous. No Python, host callbacks, or transfers in time loop.
struct RunStats {
    double milliseconds;
    std::uint64_t h2d_bytes;
    std::uint64_t d2h_bytes;
    std::uint64_t stepping_transfers;
};
int fdtd_device_count();
int fdtd_run(int precision, int nx, int ny, int nt, int ns, int nr,
    const void* ca, const void* cbx, const void* cby,
    const void* chx, const void* chy, const unsigned char* pec,
    const int* sources, const void* waveforms, const int* receivers,
    void* ez, void* hx, void* hy, void* history,
    RunStats* stats, char* error, int error_size);
