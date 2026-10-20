#include "go2_fusion_cpp/cuda_visibility.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace go2_fusion {
namespace {
struct SparsePoint { uint32_t pixel; float z; };
void check(cudaError_t error,const char* operation) {
  if(error!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
__global__ void clear_depth(float* depth,size_t n) {
  const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)depth[i]=CUDART_INF_F;
}
__global__ void rasterize(const SparsePoint* points,size_t n,float* depth) {
  const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)atomicMin(reinterpret_cast<unsigned int*>(depth+points[i].pixel),__float_as_uint(points[i].z));
}
__global__ void horizontal_min(const float* input,float* output,uint32_t w,uint32_t h,int radius) {
  // Coalesced tile loads share the halo instead of rereading it for each pixel.
  extern __shared__ float tile[];
  const int tile_width=int(blockDim.x)+2*radius;
  const int thread=int(threadIdx.y*blockDim.x+threadIdx.x);
  for(int i=thread;i<tile_width*int(blockDim.y);i+=int(blockDim.x*blockDim.y)) {
    const int x=int(blockIdx.x*blockDim.x)+i%tile_width-radius;
    const uint32_t y=blockIdx.y*blockDim.y+i/tile_width;
    tile[i]=(x>=0&&x<int(w)&&y<h)?input[y*w+x]:CUDART_INF_F;
  }
  __syncthreads();
  const uint32_t x=blockIdx.x*blockDim.x+threadIdx.x;
  const uint32_t y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=w||y>=h)return;
  float value=CUDART_INF_F;
  const int base=int(threadIdx.y)*tile_width+int(threadIdx.x);
  for(int j=0;j<=2*radius;++j)value=fminf(value,tile[base+j]);
  output[y*w+x]=value;
}
__global__ void mark_vertical(const SparsePoint* points,size_t n,const float* depth,
                             const float* horizontal,uint32_t w,uint32_t h,int radius,
                             double margin,uint8_t* flags) {
  const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {
    const SparsePoint p=points[i];
    if(p.z!=depth[p.pixel]) {flags[i]=0;return;}
    const int x=p.pixel%w,y=p.pixel/w;
    float nearest=CUDART_INF_F;
    for(int row=max(0,y-radius);row<=min(int(h)-1,y+radius);++row)
      nearest=fminf(nearest,horizontal[row*w+x]);
    // Match the CPU's double promotion of its configured margin exactly.
    flags[i]=double(p.z)<=double(nearest)+margin;
  }
}
__global__ void mark_sparse(const SparsePoint* points,size_t n,const float* depth,
                           uint32_t w,uint32_t h,int radius,double margin,uint8_t* flags) {
  // One complete warp per point. Only projected pixels need a neighborhood;
  // sparse single sweeps need no full-image local-minimum intermediates.
  const size_t i=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  const int lane=threadIdx.x%32;
  if(i>=n)return;
  const SparsePoint p=points[i];
  if(p.z!=depth[p.pixel]) {if(lane==0)flags[i]=0;return;}
  const int x=p.pixel%w,y=p.pixel/w;
  const int first=max(0,x-radius),last=min(int(w)-1,x+radius);
  float nearest=CUDART_INF_F;
  for(int row=max(0,y-radius);row<=min(int(h)-1,y+radius);++row)
    for(int col=first+lane;col<=last;col+=32)
      nearest=fminf(nearest,depth[row*w+col]);
  for(int offset=16;offset;offset/=2)
    nearest=fminf(nearest,__shfl_down_sync(0xffffffff,nearest,offset));
  if(lane==0)flags[i]=double(p.z)<=double(nearest)+margin;
}
}
struct CudaVisibility::Impl {
  Config cfg;
  TransferMode mode;
  SparsePoint* staging=nullptr;
  SparsePoint* device_input=nullptr;
  uint8_t* host_flags=nullptr;
  uint8_t* device_flags=nullptr;
  float* depth=nullptr;
  float* scratch=nullptr;
  cudaStream_t stream=nullptr;
  cudaEvent_t events[4]={};
  size_t count=0;
  std::string last_error;
  Stats timing;
  Impl(Config config,TransferMode transfer):cfg(config),mode(transfer) {}
  ~Impl() {
    if(stream)cudaStreamSynchronize(stream);
    for(auto event:events)if(event)cudaEventDestroy(event);
    if(device_input&&mode==TransferMode::ExplicitCopy)cudaFree(device_input);
    if(staging)cudaFreeHost(staging);
    if(host_flags)cudaFreeHost(host_flags);
    if(device_flags)cudaFree(device_flags);
    if(depth)cudaFree(depth);
    if(scratch)cudaFree(scratch);
    if(stream)cudaStreamDestroy(stream);
  }
};
CudaVisibility::CudaVisibility(Config config,TransferMode mode):impl_(new Impl(config,mode)) {
  // Validate the same bounds as the reference without retaining its buffers.
  if(!config.max_points||config.max_points>10000000||!config.max_width||!config.max_height||
     uint64_t(config.max_width)*config.max_height>16000000||config.occlusion_radius_px<0||
     config.occlusion_radius_px>128||!std::isfinite(config.occlusion_margin_m)||config.occlusion_margin_m<0)
    throw std::invalid_argument("invalid CUDA visibility capacities/configuration");
  auto& s=*impl_; int device=0; check(cudaGetDevice(&device),"get device");
  cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties,device),"device properties");
  if(mode==TransferMode::MappedPinned&&!properties.canMapHostMemory)
    throw std::runtime_error("device does not support mapped pinned input");
  check(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking),"create nonblocking stream");
  for(auto& event:s.events)check(cudaEventCreate(&event),"create timing event");
  const auto input_bytes=config.max_points*sizeof(SparsePoint);
  check(cudaHostAlloc(reinterpret_cast<void**>(&s.staging),input_bytes,
                      mode==TransferMode::MappedPinned?cudaHostAllocMapped:cudaHostAllocDefault),"allocate pinned input");
  if(mode==TransferMode::MappedPinned)
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s.device_input),s.staging,0),"map pinned input");
  else check(cudaMalloc(reinterpret_cast<void**>(&s.device_input),input_bytes),"allocate device input");
  check(cudaHostAlloc(reinterpret_cast<void**>(&s.host_flags),config.max_points,cudaHostAllocDefault),"allocate pinned flags");
  check(cudaMalloc(reinterpret_cast<void**>(&s.device_flags),config.max_points),"allocate device flags");
  const auto raster_bytes=size_t(config.max_width)*config.max_height*sizeof(float);
  check(cudaMalloc(reinterpret_cast<void**>(&s.depth),raster_bytes),"allocate depth");
  check(cudaMalloc(reinterpret_cast<void**>(&s.scratch),raster_bytes),"allocate horizontal minima");
  std::fill_n(s.host_flags,config.max_points,uint8_t(0));
}
CudaVisibility::~CudaVisibility()=default;
bool CudaVisibility::run(const std::vector<Projected>& points,uint32_t w,uint32_t h,bool inject_failure) {
  auto& s=*impl_; const auto start=std::chrono::steady_clock::now();
  s.timing={};s.last_error.clear();s.count=std::min(points.size(),s.cfg.max_points);
  std::fill_n(s.host_flags,s.count,uint8_t(0));
  try {
    if(inject_failure)throw std::runtime_error("injected CUDA backend failure");
    if(points.size()>s.cfg.max_points||!w||!h||w>s.cfg.max_width||h>s.cfg.max_height)
      throw std::runtime_error("CUDA visibility capacity exceeded");
    const size_t pixels=size_t(w)*h;
    for(size_t i=0;i<points.size();++i) {
      const auto& p=points[i];
      if(p.pixel>=pixels||!std::isfinite(p.camera_point.z)||p.camera_point.z<=0)
        throw std::runtime_error("invalid CUDA projected point");
      s.staging[i]={p.pixel,p.camera_point.z};
    }
    // Empty input has no flags and must not launch full-image work.
    if(!s.count) {
      s.timing.total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      return true;
    }
    check(cudaEventRecord(s.events[0],s.stream),"record start");
    if(s.mode==TransferMode::ExplicitCopy&&s.count)
      check(cudaMemcpyAsync(s.device_input,s.staging,s.count*sizeof(SparsePoint),cudaMemcpyHostToDevice,s.stream),"upload sparse points");
    check(cudaEventRecord(s.events[1],s.stream),"record upload");
    clear_depth<<<(pixels+255)/256,256,0,s.stream>>>(s.depth,pixels);
    check(cudaPeekAtLastError(),"launch clear depth");
    if(s.count) {
      rasterize<<<(s.count+255)/256,256,0,s.stream>>>(s.device_input,s.count,s.depth);
      check(cudaPeekAtLastError(),"launch rasterizer");
    }
    const int radius=s.cfg.occlusion_radius_px;
    if(s.count*size_t(2*radius+1)<=pixels) {
      mark_sparse<<<(s.count+7)/8,256,0,s.stream>>>(s.device_input,s.count,s.depth,w,h,radius,s.cfg.occlusion_margin_m,s.device_flags);
      check(cudaPeekAtLastError(),"launch sparse visibility");
    } else {
      const dim3 block(32,8),grid((w+31)/32,(h+7)/8);
      const size_t shared_bytes=size_t(32+2*radius)*8*sizeof(float);
      horizontal_min<<<grid,block,shared_bytes,s.stream>>>(s.depth,s.scratch,w,h,radius);
      check(cudaPeekAtLastError(),"launch horizontal minima");
      mark_vertical<<<(s.count+255)/256,256,0,s.stream>>>(s.device_input,s.count,s.depth,s.scratch,w,h,radius,s.cfg.occlusion_margin_m,s.device_flags);
      check(cudaPeekAtLastError(),"launch sparse vertical minima");
    }
    check(cudaEventRecord(s.events[2],s.stream),"record kernels");
    if(s.count)check(cudaMemcpyAsync(s.host_flags,s.device_flags,s.count,cudaMemcpyDeviceToHost,s.stream),"download sparse flags");
    check(cudaEventRecord(s.events[3],s.stream),"record download");
    check(cudaStreamSynchronize(s.stream),"complete visibility stream");
    float ms=0;
    check(cudaEventElapsedTime(&ms,s.events[0],s.events[1]),"time upload");s.timing.upload_ms=ms;
    check(cudaEventElapsedTime(&ms,s.events[1],s.events[2]),"time kernels");s.timing.kernels_ms=ms;
    check(cudaEventElapsedTime(&ms,s.events[2],s.events[3]),"time download");s.timing.download_ms=ms;
  } catch(const std::exception& e) {
    // Drain queued work before clearing the pinned destination. A failed CUDA
    // context may remain unusable. The owner must keep this frame invalid;
    // selecting CPU operation requires an explicit restart/backend choice.
    if(s.stream)cudaStreamSynchronize(s.stream);
    std::fill_n(s.host_flags,s.count,uint8_t(0));s.last_error=e.what();
  }
  s.timing.total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  return s.last_error.empty();
}
const uint8_t* CudaVisibility::flags() const{return impl_->host_flags;}
size_t CudaVisibility::size() const{return impl_->count;}
size_t CudaVisibility::allocated_bytes() const {
  const auto& s=*impl_;
  return size_t(s.cfg.max_width)*s.cfg.max_height*2*sizeof(float)+
    s.cfg.max_points*(sizeof(SparsePoint)*(s.mode==TransferMode::ExplicitCopy?2:1)+2);
}
const std::string& CudaVisibility::error() const{return impl_->last_error;}
const CudaVisibility::Stats& CudaVisibility::stats() const{return impl_->timing;}
}  // namespace go2_fusion
