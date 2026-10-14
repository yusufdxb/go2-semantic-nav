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
template<bool Horizontal>
__global__ void local_min(const float* input,float* output,uint32_t w,uint32_t h,int radius) {
  const uint32_t x=blockIdx.x*blockDim.x+threadIdx.x;
  const uint32_t y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=w||y>=h)return;
  const int center=Horizontal?x:y;
  const int limit=Horizontal?w:h;
  const int start=max(0,center-radius),end=min(limit-1,center+radius);
  float value=CUDART_INF_F;
  for(int j=start;j<=end;++j) {
    const uint32_t index=Horizontal?y*w+j:j*w+x;
    value=fminf(value,input[index]);
  }
  output[y*w+x]=value;
}
__global__ void mark_visible(const SparsePoint* points,size_t n,const float* depth,
                             const float* nearest,double margin,uint8_t* flags) {
  const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {
    const SparsePoint p=points[i];
    // Match the CPU's double promotion of its configured margin exactly.
    flags[i]=p.z==depth[p.pixel] && double(p.z)<=double(nearest[p.pixel])+margin;
  }
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
  float* nearest=nullptr;
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
    if(nearest)cudaFree(nearest);
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
  check(cudaMalloc(reinterpret_cast<void**>(&s.nearest),raster_bytes),"allocate visibility minima");
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
    const dim3 block(32,8),grid((w+31)/32,(h+7)/8);
    local_min<true><<<grid,block,0,s.stream>>>(s.depth,s.scratch,w,h,s.cfg.occlusion_radius_px);
    check(cudaPeekAtLastError(),"launch horizontal minima");
    local_min<false><<<grid,block,0,s.stream>>>(s.scratch,s.nearest,w,h,s.cfg.occlusion_radius_px);
    check(cudaPeekAtLastError(),"launch vertical minima");
    if(s.count) {
      mark_visible<<<(s.count+255)/256,256,0,s.stream>>>(s.device_input,s.count,s.depth,s.nearest,s.cfg.occlusion_margin_m,s.device_flags);
      check(cudaPeekAtLastError(),"launch visibility flags");
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
  return size_t(s.cfg.max_width)*s.cfg.max_height*3*sizeof(float)+
    s.cfg.max_points*(sizeof(SparsePoint)*(s.mode==TransferMode::ExplicitCopy?2:1)+2);
}
const std::string& CudaVisibility::error() const{return impl_->last_error;}
const CudaVisibility::Stats& CudaVisibility::stats() const{return impl_->timing;}
}  // namespace go2_fusion
