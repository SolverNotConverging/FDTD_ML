#include "fdtd_api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Resources {
    std::vector<void*> buffers;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    bool stepping = false;
    RunStats* stats;
    explicit Resources(RunStats* s) : stats(s) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto ptr : buffers) cudaFree(ptr);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
    }
    template<class T> T* allocate(size_t count, const T* host = nullptr) {
        if (!count) return nullptr;
        T* ptr = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&ptr), count*sizeof(T)));
        buffers.push_back(ptr);
        if (host) {
            check(cudaMemcpyAsync(ptr, host, count*sizeof(T), cudaMemcpyHostToDevice, stream));
            stats->h2d_bytes += count*sizeof(T);
            if (stepping) ++stats->stepping_transfers;
        }
        return ptr;
    }
    template<class T> void download(T* host, T* device, size_t count) {
        if (!count) return;
        check(cudaMemcpyAsync(host, device, count*sizeof(T), cudaMemcpyDeviceToHost, stream));
        stats->d2h_bytes += count*sizeof(T);
        if (stepping) ++stats->stepping_transfers;
    }
};

template<class T> __global__ void update_h(const T* e, T* hx, T* hy,
    const T* chx, const T* chy, int nx, int ny) {
    size_t p = blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (p < size_t(nx+1)*ny) {
        size_t i=p/ny, j=p%ny, eidx=i*(ny+1)+j;
        hx[p] -= chx[p]*(e[eidx+1]-e[eidx]);
    }
    if (p < size_t(nx)*(ny+1))
        hy[p] += chy[p]*(e[p+ny+1]-e[p]);
}
template<class T> __global__ void update_e(T* e, const T* hx, const T* hy,
    const T* ca, const T* cbx, const T* cby, const unsigned char* pec, int nx, int ny) {
    size_t p = blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    if (p >= size_t(nx+1)*(ny+1)) return;
    int i=int(p/(ny+1)), j=int(p%(ny+1));
    if (pec[p] || i==0 || j==0 || i==nx || j==ny) { e[p]=T(0); return; }
    e[p] = ca[p]*e[p] + cbx[p]*(hy[p]-hy[p-ny-1])
                        - cby[p]*(hx[size_t(i)*ny+j]-hx[size_t(i)*ny+j-1]);
}
// Sources are deduplicated on the host before upload, so overlapping lines/points
// sum deterministically and need no floating-point atomics.
template<class T> __global__ void inject(T* e, const int* sites, const T* waves,
    size_t offset, int ns) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if (s<ns) e[sites[s]] += waves[offset+s];
}
template<class T> __global__ void sample(const T* e, const int* sites, T* history,
    size_t offset, int nr) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if (r<nr) history[offset+r]=e[sites[r]];
}

template<class T> void run(int nx,int ny,int nt,int ns,int nr,
    const void* ca,const void* cbx,const void* cby,const void* chx,const void* chy,
    const unsigned char* pec,const int* sources,const void* waveforms,const int* receivers,
    void* ez,void* hx,void* hy,void* history,RunStats* stats) {
    Resources r(stats);
    size_t ne=size_t(nx+1)*(ny+1), nhx=size_t(nx+1)*ny, nhy=size_t(nx)*(ny+1);
    auto de=r.allocate(ne, static_cast<T*>(ez));
    auto dhx=r.allocate(nhx, static_cast<T*>(hx));
    auto dhy=r.allocate(nhy, static_cast<T*>(hy));
    auto dca=r.allocate(ne, static_cast<const T*>(ca));
    auto dcx=r.allocate(ne, static_cast<const T*>(cbx));
    auto dcy=r.allocate(ne, static_cast<const T*>(cby));
    auto dchx=r.allocate(nhx, static_cast<const T*>(chx));
    auto dchy=r.allocate(nhy, static_cast<const T*>(chy));
    auto dp=r.allocate(ne, pec);
    auto ds=r.allocate(size_t(ns), sources);
    auto dw=r.allocate(size_t(nt)*ns, static_cast<const T*>(waveforms));
    auto dr=r.allocate(size_t(nr), receivers);
    auto out=r.allocate<T>(size_t(nt)*nr);
    check(cudaEventCreate(&r.start)); check(cudaEventCreate(&r.stop));
    check(cudaEventRecord(r.start,r.stream));
    r.stepping=true;
    for(int n=0;n<nt;++n) {
        update_h<<<unsigned((std::max(nhx,nhy)+255)/256),256,0,r.stream>>>(de,dhx,dhy,dchx,dchy,nx,ny);
        update_e<<<unsigned((ne+255)/256),256,0,r.stream>>>(de,dhx,dhy,dca,dcx,dcy,dp,nx,ny);
        if(ns) inject<<<(ns+255)/256,256,0,r.stream>>>(de,ds,dw,size_t(n)*ns,ns);
        if(nr) sample<<<(nr+255)/256,256,0,r.stream>>>(de,dr,out,size_t(n)*nr,nr);
    }
    r.stepping=false;
    check(cudaGetLastError());
    check(cudaEventRecord(r.stop,r.stream));
    check(cudaEventSynchronize(r.stop));
    float ms=0; check(cudaEventElapsedTime(&ms,r.start,r.stop)); stats->milliseconds=ms;
    r.download(static_cast<T*>(ez),de,ne);
    r.download(static_cast<T*>(hx),dhx,nhx);
    r.download(static_cast<T*>(hy),dhy,nhy);
    r.download(static_cast<T*>(history),out,size_t(nt)*nr);
    check(cudaStreamSynchronize(r.stream));
}
}
int fdtd_device_count() {
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess) { cudaGetLastError(); return 0; }
    return count;
}
int fdtd_run(int precision,int nx,int ny,int nt,int ns,int nr,
    const void* ca,const void* cbx,const void* cby,const void* chx,const void* chy,
    const unsigned char* pec,const int* sources,const void* waveforms,const int* receivers,
    void* ez,void* hx,void* hy,void* history,RunStats* stats,char* error,int error_size) {
    *stats = RunStats{};
    try {
        if(precision==32) run<float>(nx,ny,nt,ns,nr,ca,cbx,cby,chx,chy,pec,sources,waveforms,receivers,ez,hx,hy,history,stats);
        else if(precision==64) run<double>(nx,ny,nt,ns,nr,ca,cbx,cby,chx,chy,pec,sources,waveforms,receivers,ez,hx,hy,history,stats);
        else throw std::runtime_error("Unsupported field precision");
        return 0;
    } catch(const std::exception& e) {
        std::snprintf(error,error_size,"%s",e.what()); return 1;
    }
}
