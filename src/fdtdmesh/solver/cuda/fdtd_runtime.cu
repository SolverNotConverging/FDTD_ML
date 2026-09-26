#include "runtime_types.h"
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr double pi=3.14159265358979323846, c0=299792458.0, eta=4e-7*pi*c0;
constexpr int threads=256;
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct State { int step, status, stable, checks; double incident_peak, error, residual; };
struct Resources {
    std::vector<void*> buffers;
    cudaStream_t work=nullptr, telemetry=nullptr;
    cudaEvent_t begin=nullptr, done=nullptr, report_done=nullptr, initialized=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
    Progress* pinned=nullptr;
    RunStats* stats;
    explicit Resources(RunStats* s):stats(s) {
        int low, high; check(cudaDeviceGetStreamPriorityRange(&low,&high));
        check(cudaStreamCreateWithPriority(&work,cudaStreamNonBlocking,high));
        check(cudaStreamCreateWithPriority(&telemetry,cudaStreamNonBlocking,low));
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&done));
        check(cudaEventCreateWithFlags(&report_done,cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&initialized,cudaEventDisableTiming));
        check(cudaMallocHost(&pinned,sizeof(Progress)));
    }
    ~Resources() {
        if(work) cudaStreamSynchronize(work);
        if(telemetry) cudaStreamSynchronize(telemetry);
        if(exec) cudaGraphExecDestroy(exec);
        if(graph) cudaGraphDestroy(graph);
        for(auto p:buffers) cudaFree(p);
        if(pinned) cudaFreeHost(pinned);
        if(begin) cudaEventDestroy(begin);
        if(done) cudaEventDestroy(done);
        if(report_done) cudaEventDestroy(report_done);
        if(initialized) cudaEventDestroy(initialized);
        if(work) cudaStreamDestroy(work);
        if(telemetry) cudaStreamDestroy(telemetry);
    }
    template<class T> T* allocate(size_t n,const T* input=nullptr) {
        T* p; check(cudaMalloc(&p,n*sizeof(T))); buffers.push_back(p);
        stats->device_bytes+=n*sizeof(T);
        if(input) { check(cudaMemcpyAsync(p,input,n*sizeof(T),cudaMemcpyHostToDevice,work)); stats->h2d_bytes+=n*sizeof(T); }
        else check(cudaMemsetAsync(p,0,n*sizeof(T),work));
        return p;
    }
    template<class T> void download(T* out,const T* in,size_t n,std::uint64_t& counter) {
        check(cudaMemcpyAsync(out,in,n*sizeof(T),cudaMemcpyDeviceToHost,work)); counter+=n*sizeof(T);
    }
};
template<class T> struct Device {
    RunConfig cfg;
    T *e,*hx,*hy,*ei,*hi,*pex,*pey,*phx,*phy,*pei,*phi;
    T *ex,*ey,*ih,*hxq,*hyq,*hxc,*hyc,*profiles;
    int *hxp,*hyp,*monitors;
    unsigned char* pec;
    double *points,*normals,*ds,*weights,*freq,*angles,*currents,*previous,*incident,*prev_inc;
    double *bin_error,*field_partial,*amplitude,*width;
    State* state;
    Progress *snapshots,*report;
    int* published;
    int field_blocks;
};
__device__ bool outside_e(int i,int j,const RunConfig& c) { return i<c.a || i>c.b || j<c.c || j>c.d; }
__device__ bool outside_y(int i,int j,const RunConfig& c) { return i<c.a || i>=c.b || j<c.c || j>c.d; }
template<class T> __device__ T raw_x(Device<T> d,int p) {
    int i=p/d.cfg.ny,j=p%d.cfg.ny,e=i*(d.cfg.ny+1)+j;
    return d.e[e+1]-d.e[e]+T(int(outside_e(i,j+1,d.cfg))-int(outside_e(i,j,d.cfg)))*d.ei[i];
}
template<class T> __device__ T raw_y(Device<T> d,int p) {
    int i=p/(d.cfg.ny+1),j=p%(d.cfg.ny+1);
    T delta=d.e[p+d.cfg.ny+1]-d.e[p];
    delta+=T(outside_e(i+1,j,d.cfg))*d.ei[i+1]-T(outside_e(i,j,d.cfg))*d.ei[i];
    return delta-T(outside_y(i,j,d.cfg))*(d.ei[i+1]-d.ei[i]);
}
template<class T> __global__ void update_h(Device<T> d) {
    if(d.state->step>=d.cfg.max_steps) return;
    int p=blockIdx.x*blockDim.x+threadIdx.x,nx=d.cfg.nx,ny=d.cfg.ny;
    if(p<(nx+1)*ny) {
        int j=p%ny; const T* c=d.profiles+3*(2*nx+ny+2+j);
        T v=raw_x(d,p); d.phx[p]=c[1]*d.phx[p]+c[2]*v;
        T change=d.hxq[p]*(c[0]*v+d.phx[p]);
        if(d.hxp[p]>=0) change+=d.hxc[p]*raw_x(d,d.hxp[p]);
        d.hx[p]-=change;
    }
    if(p<nx*(ny+1)) {
        int i=p/(ny+1); const T* c=d.profiles+3*(nx+ny+2+i);
        T v=raw_y(d,p); d.phy[p]=c[1]*d.phy[p]+c[2]*v;
        T change=d.hyq[p]*(c[0]*v+d.phy[p]);
        if(d.hyp[p]>=0) change+=d.hyc[p]*raw_y(d,d.hyp[p]);
        d.hy[p]+=change;
    }
    if(p<nx) {
        const T* c=d.profiles+3*(nx+ny+2+p);
        T v=d.ei[p+1]-d.ei[p]; d.phi[p]=c[1]*d.phi[p]+c[2]*v;
        d.hi[p]+=d.ih[p]*(c[0]*v+d.phi[p]);
    }
}
template<class T> __global__ void update_e(Device<T> d) {
    int n=d.state->step; if(n>=d.cfg.max_steps) return;
    int p=blockIdx.x*blockDim.x+threadIdx.x,nx=d.cfg.nx,ny=d.cfg.ny;
    if(p<(nx+1)*(ny+1)) {
        int i=p/(ny+1),j=p%(ny+1);
        if(d.pec[p]) d.e[p]=0;
        else {
            T vx=d.hy[p]-d.hy[p-ny-1];
            vx+=T(outside_y(i,j,d.cfg))*d.hi[i]-T(outside_y(i-1,j,d.cfg))*d.hi[i-1];
            vx-=T(outside_e(i,j,d.cfg))*(d.hi[i]-d.hi[i-1]);
            T vy=d.hx[i*ny+j]-d.hx[i*ny+j-1];
            const T* cx=d.profiles+3*i; const T* cy=d.profiles+3*(nx+1+j);
            d.pex[p]=cx[1]*d.pex[p]+cx[2]*vx; d.pey[p]=cy[1]*d.pey[p]+cy[2]*vy;
            d.e[p]+=d.ex[i]*(cx[0]*vx+d.pex[p])-d.ey[j]*(cy[0]*vy+d.pey[p]);
        }
    }
    if(p>0 && p<nx) {
        const T* c=d.profiles+3*p;
        T v=d.hi[p]-d.hi[p-1]; d.pei[p]=c[1]*d.pei[p]+c[2]*v;
        d.ei[p]+=d.ex[p]*(c[0]*v+d.pei[p]);
        if(p==d.cfg.source) {
            double u=(n+1)*d.cfg.dt*d.cfg.f0-d.cfg.pulse_delay;
            d.ei[p]+=T(d.cfg.source_scale*exp(-u*u/(d.cfg.pulse_width*d.cfg.pulse_width))*cos(2*pi*u));
        }
    }
}
template<class T> __global__ void accumulate_dft(Device<T> d) {
    int n=d.state->step; if(n>=d.cfg.max_steps) return;
    int f=blockIdx.x,tid=threadIdx.x,nr=d.cfg.nr,nx=d.cfg.nx,ny=d.cfg.ny;
    double w=2*pi*d.freq[f]*d.cfg.dt,se,ce,sh,ch;
    sincos(-w*(n+1),&se,&ce); sincos(-w*(n+.5),&sh,&ch);
    int nhx=(nx+1)*ny;
    for(int s=tid;s<nr;s+=blockDim.x) {
        int e=d.monitors[3*s],a=d.monitors[3*s+1],b=d.monitors[3*s+2];
        double ha=a<nhx?d.hx[a]:d.hy[a-nhx],hb=b<nhx?d.hx[b]:d.hy[b-nhx];
        double E=d.e[e]*d.cfg.dt,H=(ha*d.weights[2*s]+hb*d.weights[2*s+1])*d.cfg.dt;
        double* z=d.currents+4*(f*nr+s);
        z[0]+=E*ce;z[1]+=E*se;z[2]+=H*ch;z[3]+=H*sh;
    }
    for(int i=tid;i<=nx;i+=blockDim.x) {
        double E=d.ei[i]*d.cfg.dt;double* z=d.incident+2*(f*(nx+1)+i);
        z[0]+=E*ce;z[1]+=E*se;
    }
}
template<class T> __global__ void tick(Device<T> d) {
    if(d.state->step<d.cfg.max_steps) {
        d.state->incident_peak=fmax(d.state->incident_peak,fabs(double(d.ei[d.cfg.origin])));
        ++d.state->step;
    }
}
template<class T> __device__ bool checkpoint(Device<T> d) {
    return d.state->step%d.cfg.check_interval==0 || d.state->step==d.cfg.max_steps;
}
template<class T> __global__ void field_check(Device<T> d) {
    if(!checkpoint(d)) return;
    int p=blockIdx.x*blockDim.x+threadIdx.x,nx=d.cfg.nx,ny=d.cfg.ny;double v=0;
    if(p<(nx+1)*(ny+1)) {v=fabs(double(d.e[p]));if(!isfinite(v)) v=INFINITY;}
    if(p<(nx+1)*ny) { double z=eta*fabs(double(d.hx[p]));v=isfinite(z)?fmax(v,z):INFINITY; }
    if(p<nx*(ny+1)) { double z=eta*fabs(double(d.hy[p]));v=isfinite(z)?fmax(v,z):INFINITY; }
    if(!isfinite(v)) v=INFINITY;
    __shared__ double vals[threads];vals[threadIdx.x]=v;__syncthreads();
    for(int s=threads/2;s;s/=2) {if(threadIdx.x<s) vals[threadIdx.x]=fmax(vals[threadIdx.x],vals[threadIdx.x+s]);__syncthreads();}
    if(threadIdx.x==0) d.field_partial[blockIdx.x]=vals[0];
}
template<class T> __global__ void dft_check(Device<T> d) {
    if(!checkpoint(d)) return;
    int f=blockIdx.x,tid=threadIdx.x,nr=d.cfg.nr;double delta=0,norm=0,perimeter=0;
    for(int s=tid;s<nr;s+=threads) {
        for(int k=0;k<4;++k) {
            size_t p=4*size_t(f*nr+s)+k;
            double z=d.currents[p],change=z-d.previous[p],scale=k<2?1:eta*eta;
            norm+=d.ds[s]*scale*z*z;delta+=d.ds[s]*scale*change*change;d.previous[p]=z;
        }
        perimeter+=d.ds[s];
    }
    __shared__ double v[3][threads];v[0][tid]=delta;v[1][tid]=norm;v[2][tid]=perimeter;__syncthreads();
    for(int s=threads/2;s;s/=2) {if(tid<s) for(int k=0;k<3;++k) v[k][tid]+=v[k][tid+s];__syncthreads();}
    if(tid==0) {
        const double* inc=d.incident+2*(f*(d.cfg.nx+1)+d.cfg.origin);
        double strength=hypot(inc[0],inc[1]),change=hypot(inc[0]-d.prev_inc[2*f],inc[1]-d.prev_inc[2*f+1]);
        d.prev_inc[2*f]=inc[0];d.prev_inc[2*f+1]=inc[1];
        double tolerance=d.cfg.atol*strength*sqrt(v[2][0])+d.cfg.rtol*sqrt(v[1][0]);
        double error=sqrt(v[0][0])/fmax(tolerance,1e-300);
        error=fmax(error,change/fmax((d.cfg.atol+d.cfg.rtol)*strength,1e-300));
        if(!isfinite(v[0][0]) || !isfinite(v[1][0]) || strength<1e-20 || !isfinite(strength)) error=INFINITY;
        d.bin_error[f]=error;
    }
}
template<class T> __global__ void decide(Device<T> d,cudaGraphConditionalHandle handle) {
    if(checkpoint(d)) {
        double error=0,residual=0;
        for(int f=0;f<d.cfg.nf;++f) error=fmax(error,d.bin_error[f]);
        for(int b=0;b<d.field_blocks;++b) residual=fmax(residual,d.field_partial[b]);
        residual/=fmax(d.state->incident_peak,1e-30);
        bool eligible=d.state->step>=d.cfg.min_steps;
        d.state->stable=(eligible && error<1 && residual<d.cfg.field_tol)?d.state->stable+1:0;
        if(!isfinite(residual)) d.state->status=3;
        else if(d.cfg.auto_stop && d.state->stable>=d.cfg.stable_checks) d.state->status=1;
        else if(d.state->step>=d.cfg.max_steps) d.state->status=2;
        d.state->error=error;d.state->residual=residual;int generation=++d.state->checks;
        d.snapshots[generation-1]={generation,d.state->step,d.state->status,d.state->stable,
                                  d.state->step*d.cfg.dt,error,residual,d.state->incident_peak};
        // Immutable entries, published only after completion. Telemetry never races a writer.
        cuda::atomic_ref<int,cuda::thread_scope_device>(*d.published).store(generation,cuda::memory_order_release);
    }
    cudaGraphSetConditional(handle,d.state->status==0?1:0);
}
template<class T> __global__ void get_report(Device<T> d) {
    int n=cuda::atomic_ref<int,cuda::thread_scope_device>(*d.published).load(cuda::memory_order_acquire);
    *d.report=n?d.snapshots[n-1]:Progress{};
}
template<class T> __global__ void near_to_far(Device<T> d) {
    int a=blockIdx.x,f=blockIdx.y,tid=threadIdx.x;
    double ux=cos(d.angles[a]),uy=sin(d.angles[a]),k=2*pi*d.freq[f]/c0,re=0,im=0;
    for(int s=tid;s<d.cfg.nr;s+=threads) {
        const double* z=d.currents+4*(f*d.cfg.nr+s);
        double dot=ux*d.normals[2*s]+uy*d.normals[2*s+1];
        double ar=dot*z[0]-eta*z[2],ai=dot*z[1]-eta*z[3],sn,cs;
        sincos(k*(ux*(d.points[2*s]-d.cfg.origin_x)+uy*(d.points[2*s+1]-d.cfg.origin_y)),&sn,&cs);
        re+=(ar*cs-ai*sn)*d.ds[s];im+=(ar*sn+ai*cs)*d.ds[s];
    }
    __shared__ double v[2][threads];v[0][tid]=re;v[1][tid]=im;__syncthreads();
    for(int s=threads/2;s;s/=2) {if(tid<s) {v[0][tid]+=v[0][tid+s];v[1][tid]+=v[1][tid+s];}__syncthreads();}
    if(tid==0) {
        const double* inc=d.incident+2*(f*(d.cfg.nx+1)+d.cfg.origin);
        double norm=inc[0]*inc[0]+inc[1]*inc[1],scale=k/4/norm;
        re=scale*(v[0][0]*inc[0]+v[1][0]*inc[1]);im=scale*(v[1][0]*inc[0]-v[0][0]*inc[1]);
        d.amplitude[2*(f*d.cfg.nd+a)]=re;d.amplitude[2*(f*d.cfg.nd+a)+1]=im;
        d.width[f*d.cfg.nd+a]=4/k*(re*re+im*im);
    }
}
cudaGraphNode_t kernel(cudaGraph_t graph,cudaGraphNode_t prior,void* func,dim3 blocks,dim3 block,void** args) {
    cudaKernelNodeParams p{};p.func=func;p.gridDim=blocks;p.blockDim=block;p.kernelParams=args;
    cudaGraphNode_t node;check(cudaGraphAddKernelNode(&node,graph,prior?&prior:nullptr,prior?1:0,&p));return node;
}
template<class T> void run(const RunConfig& c,const HostInputs& in,HostOutputs& out,RunStats* stats,
                          ProgressCallback callback,void* context) {
    Resources r(stats);Device<T> d{};d.cfg=c;
    size_t ne=size_t(c.nx+1)*(c.ny+1),nhx=size_t(c.nx+1)*c.ny,nhy=size_t(c.nx)*(c.ny+1);
    d.e=r.allocate(ne,static_cast<const T*>(in.initial_e));d.hx=r.allocate(nhx,static_cast<const T*>(in.initial_hx));
    d.hy=r.allocate(nhy,static_cast<const T*>(in.initial_hy));d.ei=r.allocate<T>(c.nx+1);d.hi=r.allocate<T>(c.nx);
    d.pex=r.allocate<T>(ne);d.pey=r.allocate<T>(ne);d.phx=r.allocate<T>(nhx);d.phy=r.allocate<T>(nhy);
    d.pei=r.allocate<T>(c.nx+1);d.phi=r.allocate<T>(c.nx);
    d.ex=r.allocate(c.nx+1,static_cast<const T*>(in.ex));d.ey=r.allocate(c.ny+1,static_cast<const T*>(in.ey));
    d.ih=r.allocate(c.nx,static_cast<const T*>(in.ih));
    d.hxq=r.allocate(nhx,static_cast<const T*>(in.hxq));d.hyq=r.allocate(nhy,static_cast<const T*>(in.hyq));
    d.hxc=r.allocate(nhx,static_cast<const T*>(in.hxc));d.hyc=r.allocate(nhy,static_cast<const T*>(in.hyc));
    d.hxp=r.allocate(nhx,in.hxp);d.hyp=r.allocate(nhy,in.hyp);d.pec=r.allocate(ne,in.pec);
    d.profiles=r.allocate(3*(2*c.nx+2*c.ny+2),static_cast<const T*>(in.profiles));
    d.monitors=r.allocate(3*c.nr,in.monitors);d.weights=r.allocate(2*c.nr,in.weights);
    d.points=r.allocate(2*c.nr,in.points);d.normals=r.allocate(2*c.nr,in.normals);d.ds=r.allocate(c.nr,in.ds);
    d.freq=r.allocate(c.nf,in.frequencies);d.angles=r.allocate(c.nd,in.angles);
    d.currents=r.allocate<double>(4*size_t(c.nf)*c.nr);d.previous=r.allocate<double>(4*size_t(c.nf)*c.nr);
    d.incident=r.allocate<double>(2*size_t(c.nf)*(c.nx+1));d.prev_inc=r.allocate<double>(2*c.nf);
    d.bin_error=r.allocate<double>(c.nf);d.field_blocks=int((ne+threads-1)/threads);
    d.field_partial=r.allocate<double>(d.field_blocks);
    d.amplitude=r.allocate<double>(2*size_t(c.nf)*c.nd);d.width=r.allocate<double>(size_t(c.nf)*c.nd);
    d.state=r.allocate<State>(1);d.published=r.allocate<int>(1);
    d.snapshots=r.allocate<Progress>(c.max_steps/c.check_interval+2);d.report=r.allocate<Progress>(1);
    check(cudaEventRecord(r.initialized,r.work));
    check(cudaStreamWaitEvent(r.telemetry,r.initialized,0));
    check(cudaGraphCreate(&r.graph,0));cudaGraphConditionalHandle condition;
    check(cudaGraphConditionalHandleCreate(&condition,r.graph,1,cudaGraphCondAssignDefault));
    cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=condition;
    p.conditional.type=cudaGraphCondTypeWhile;p.conditional.size=1;cudaGraphNode_t loop;
    check(cudaGraphAddNode(&loop,r.graph,nullptr,nullptr,0,&p));
    cudaGraph_t body=p.conditional.phGraph_out[0];cudaGraphNode_t last=nullptr;void* args[]={&d};
    int batch=c.check_interval%16==0?16:1;
    for(int n=0;n<batch;++n) {
        last=kernel(body,last,(void*)update_h<T>,d.field_blocks,threads,args);
        last=kernel(body,last,(void*)update_e<T>,d.field_blocks,threads,args);
        last=kernel(body,last,(void*)accumulate_dft<T>,c.nf,threads,args);
        last=kernel(body,last,(void*)tick<T>,1,1,args);
    }
    last=kernel(body,last,(void*)field_check<T>,d.field_blocks,threads,args);
    last=kernel(body,last,(void*)dft_check<T>,c.nf,threads,args);
    void* decide_args[]={&d,&condition};kernel(body,last,(void*)decide<T>,1,1,decide_args);
    kernel(r.graph,loop,(void*)near_to_far<T>,dim3(c.nd,c.nf),threads,args);
    check(cudaGraphInstantiate(&r.exec,r.graph,0));
    check(cudaEventRecord(r.begin,r.work));check(cudaGraphLaunch(r.exec,r.work));check(cudaEventRecord(r.done,r.work));
    bool pending=false;int delivered=0;auto next=std::chrono::steady_clock::now();
    while(true) {
        cudaError_t status=cudaEventQuery(r.done);
        if(status==cudaSuccess) break;
        if(status!=cudaErrorNotReady) check(status);
        if(pending) {
            status=cudaEventQuery(r.report_done);
            if(status==cudaSuccess) {
                pending=false;
                if(callback && r.pinned->generation>delivered) {delivered=r.pinned->generation;callback(r.pinned,context);}
            } else if(status!=cudaErrorNotReady) check(status);
        }
        if(callback && !pending && std::chrono::steady_clock::now()>=next) {
            // No event or dependency points back to the FDTD stream.
            get_report<<<1,1,0,r.telemetry>>>(d);
            check(cudaMemcpyAsync(r.pinned,d.report,sizeof(Progress),cudaMemcpyDeviceToHost,r.telemetry));
            stats->telemetry_bytes+=sizeof(Progress);check(cudaEventRecord(r.report_done,r.telemetry));pending=true;
            next=std::chrono::steady_clock::now()+std::chrono::milliseconds(100);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Final retrieval only: never used to decide convergence or advance the GPU.
    check(cudaStreamSynchronize(r.telemetry));
    float ms;check(cudaEventElapsedTime(&ms,r.begin,r.done));stats->gpu_ms=ms;
    State state{};r.download(&state,d.state,1,stats->telemetry_bytes);
    r.download(out.amplitude,d.amplitude,2*size_t(c.nf)*c.nd,stats->result_bytes);
    r.download(out.width,d.width,size_t(c.nf)*c.nd,stats->result_bytes);
    if(c.debug) {
        r.download(static_cast<T*>(out.ez),d.e,ne,stats->debug_bytes);
        r.download(static_cast<T*>(out.hx),d.hx,nhx,stats->debug_bytes);r.download(static_cast<T*>(out.hy),d.hy,nhy,stats->debug_bytes);
        r.download(out.currents,d.currents,4*size_t(c.nf)*c.nr,stats->debug_bytes);
        r.download(out.incident,d.incident,2*size_t(c.nf)*(c.nx+1),stats->debug_bytes);
    }
    check(cudaStreamSynchronize(r.work));
    std::vector<Progress> records(state.checks);r.download(records.data(),d.snapshots,records.size(),stats->telemetry_bytes);
    check(cudaStreamSynchronize(r.work));
    for(int i=0;i<state.checks;++i) {
        const auto& v=records[i];double* dst=out.history+8*i;
        dst[0]=v.step;dst[1]=v.time;dst[2]=v.error;dst[3]=v.residual;dst[4]=v.stable;dst[5]=v.status;dst[6]=v.incident_peak;dst[7]=v.generation;
    }
    if(callback && !records.empty() && records.back().generation>delivered) callback(&records.back(),context);
    stats->steps=state.step;stats->status=state.status;stats->checks=state.checks;stats->reports=state.checks;
    stats->stable=state.stable;stats->error=state.error;stats->residual=state.residual;
}
}
int fdtd_device_count() {int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess){cudaGetLastError();return 0;}return n;}
int fdtd_run(int precision,const RunConfig* cfg,const HostInputs* in,HostOutputs* out,RunStats* stats,
             ProgressCallback cb,void* ctx,char* error,int size) {
    try {
        *stats={};
        if(precision==32) run<float>(*cfg,*in,*out,stats,cb,ctx);
        else if(precision==64) run<double>(*cfg,*in,*out,stats,cb,ctx);
        else throw std::runtime_error("Unsupported precision");
        return 0;
    } catch(const std::exception& e) {std::snprintf(error,size,"%s",e.what());return 1;}
}
