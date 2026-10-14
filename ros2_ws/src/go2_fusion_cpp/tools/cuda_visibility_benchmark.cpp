#include "go2_fusion_cpp/cuda_visibility.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sys/resource.h>

using namespace go2_fusion;
namespace {
template<class T> void read(std::istream& in,T& value) {
  if(!in.read(reinterpret_cast<char*>(&value),sizeof(value)))throw std::runtime_error("truncated replay");
}
struct Samples {
  std::vector<double> cpu,total,upload,kernels,download,full_cpu,full_cuda;
  void add(double reference,const CudaVisibility::Stats& s) {
    cpu.push_back(reference);total.push_back(s.total_ms);upload.push_back(s.upload_ms);
    kernels.push_back(s.kernels_ms);download.push_back(s.download_ms);
  }
  static double quantile(std::vector<double> v,double p) {
    if(v.empty())return 0;std::sort(v.begin(),v.end());return v[size_t(std::ceil(p*v.size()))-1];
  }
  static void print(const std::vector<double>& v) {
    std::cout<<"{\"p50_ms\":"<<quantile(v,.5)<<",\"p95_ms\":"<<quantile(v,.95)
      <<",\"p99_ms\":"<<quantile(v,.99)<<",\"max_ms\":"<<quantile(v,1)<<"}";
  }
};
bool equal_number(double a,double b) {return (std::isnan(a)&&std::isnan(b))||a==b;}
void measurement_parity(const Engine& cpu,const Engine& cuda,size_t& comparisons) {
  if(cpu.measurements().size()!=cuda.measurements().size())throw std::runtime_error("measurement count parity failed");
  for(size_t i=0;i<cpu.measurements().size();++i) {
    const auto& a=cpu.measurements()[i];const auto& b=cuda.measurements()[i];++comparisons;
    if(a.status!=b.status||a.label_id!=b.label_id||a.support!=b.support||a.track_id!=b.track_id||a.stamp_ns!=b.stamp_ns||
       !equal_number(a.position.x,b.position.x)||!equal_number(a.position.y,b.position.y)||!equal_number(a.position.z,b.position.z))
      throw std::runtime_error("object measurement/track parity failed");
    for(size_t j=0;j<a.covariance.size();++j)if(!equal_number(a.covariance[j],b.covariance[j]))
      throw std::runtime_error("object covariance parity failed");
  }
}
size_t rss_kib() {
  std::ifstream in("/proc/self/status");std::string line;
  while(std::getline(in,line))if(line.rfind("VmRSS:",0)==0)return std::stoull(line.substr(6));
  return 0;
}
void parity(CudaVisibility& gpu,const std::vector<Projected>& projected,uint32_t w,uint32_t h,
            size_t& comparisons,size_t& mismatches) {
  if(!gpu.run(projected,w,h))throw std::runtime_error(gpu.error());
  if(gpu.size()!=projected.size())throw std::runtime_error("CUDA flag count mismatch");
  for(size_t i=0;i<projected.size();++i) {
    ++comparisons;if(bool(gpu.flags()[i])!=projected[i].visible)++mismatches;
  }
}
void faults(CudaVisibility& gpu,const Config& cfg) {
  std::vector<Projected> points(1);points[0].camera_point={0,0,2};points[0].pixel=0;
  if(gpu.run(points,64,48,true)||gpu.flags()[0]!=0)throw std::runtime_error("injected failure did not invalidate flags");
  if(!gpu.run(points,64,48)||gpu.flags()[0]!=1)throw std::runtime_error("injected failure recovery failed");
  points[0].pixel=64*48;
  if(gpu.run(points,64,48)||gpu.flags()[0]!=0)throw std::runtime_error("invalid pixel did not fail closed");
  points[0].pixel=0;points[0].camera_point.z=NAN;
  if(gpu.run(points,64,48)||gpu.flags()[0]!=0)throw std::runtime_error("nonfinite depth did not fail closed");
  points.assign(cfg.max_points+1,Projected{});
  if(gpu.run(points,64,48)||gpu.size()>cfg.max_points)throw std::runtime_error("capacity failure unbounded");
  for(size_t i=0;i<gpu.size();++i)if(gpu.flags()[i])throw std::runtime_error("capacity failure retained valid flag");
  points.clear();if(!gpu.run(points,64,48)||gpu.size()!=0)throw std::runtime_error("empty cloud handling failed");
}
void synthetic_parity(CudaVisibility& gpu,size_t& comparisons,size_t& mismatches) {
  std::mt19937 random(7291);Config cfg;Engine reference(cfg);
  for(int scene=0;scene<16;++scene) {
    Camera cam;cam.width=64+scene;cam.height=48+scene;cam.fx=40;cam.fy=40;cam.cx=32;cam.cy=24;
    Context ctx;ctx.image_ns=1000000000LL+scene*100000000LL;ctx.cloud_ns=ctx.image_ns;ctx.tf_valid=true;
    std::vector<Point> points;
    for(int j=0;j<2000;++j) {
      const double u=random()%cam.width,v=random()%cam.height,z=0.5+(random()%650)*0.01;
      points.push_back({float((u-cam.cx)*z/cam.fx),float((v-cam.cy)*z/cam.fy),float(z)});
      if(j%20==0)points.push_back(points.back());
    }
    reference.process(cam,ctx,points.data(),points.size(),nullptr,0);
    parity(gpu,reference.projected(),cam.width,cam.height,comparisons,mismatches);
  }
}
void synthetic_core_parity(CudaVisibility& gpu,size_t& comparisons) {
  Config cfg;Engine cpu(cfg);Engine cuda(cfg,[&](std::vector<Projected>& p,uint32_t w,uint32_t h) {
    if(!gpu.run(p,w,h))return false;
    for(size_t i=0;i<p.size();++i)p[i].visible=gpu.flags()[i]!=0;
    return true;
  });
  Camera cam;cam.width=64;cam.height=48;cam.fx=cam.fy=40;cam.cx=32;cam.cy=24;
  Context ctx;ctx.image_ns=1000000000;ctx.cloud_ns=ctx.image_ns;ctx.tf_valid=true;
  ctx.intrinsics_trusted=ctx.extrinsics_trusted=ctx.timing_trusted=true;
  ctx.output_from_camera.rotation={0,-1,0,1,0,0,0,0,1};ctx.output_from_camera.translation={1,2,3};
  std::vector<Point> points;
  for(int v:{8,24,40})for(int u:{8,24,40})points.push_back({float((u-32)*2.0/40),float((v-24)*2.0/40),2});
  Detection d;d.x1=64;d.y1=48;d.score=.9F;d.label_id=7;
  auto exercise=[&](const std::vector<Detection>& detections) {
    const auto a=cpu.process(cam,ctx,points.data(),points.size(),detections.data(),detections.size());
    const auto b=cuda.process(cam,ctx,points.data(),points.size(),detections.data(),detections.size());
    if(a!=b)throw std::runtime_error("synthetic full-core status mismatch");
    measurement_parity(cpu,cuda,comparisons);ctx.image_ns+=100000000;ctx.cloud_ns=ctx.image_ns;
  };
  exercise({d});
  if(cpu.measurements()[0].status!=Status::Valid||cpu.measurements()[0].support!=9)
    throw std::runtime_error("synthetic valid object missing");
  exercise({d});exercise({});
  std::vector<uint8_t> mask(40*40,0);
  for(int v:{8,24})for(int u:{8,24,40})mask[(v-8)*40+u-8]=1;
  d.mask=mask.data();d.mask_size=mask.size();d.mask_x=8;d.mask_y=8;d.mask_width=40;d.mask_height=40;
  exercise({d});
  if(cpu.measurements()[0].status!=Status::Valid||cpu.measurements()[0].support!=6)
    throw std::runtime_error("synthetic ROI object missing");
  auto other=d;other.label_id=8;exercise({d,other});
  d.score=.99F;other.score=.7F;exercise({d,other});
  ctx.tf_valid=false;exercise({d});ctx.tf_valid=true;
  ctx.intrinsics_trusted=false;exercise({d});ctx.intrinsics_trusted=true;
  ctx.cloud_ns=ctx.image_ns-50000001;exercise({d});
}
void benchmark(const char* replay,int repeats,CudaVisibility::TransferMode mode) {
  Config cfg;Camera camera;Engine reference(cfg);CudaVisibility gpu(cfg,mode);Samples samples;
  Engine accelerated(cfg,[&](std::vector<Projected>& projected,uint32_t w,uint32_t h) {
    if(!gpu.run(projected,w,h))return false;
    for(size_t i=0;i<projected.size();++i)projected[i].visible=gpu.flags()[i]!=0;
    return true;
  });
  std::vector<Point> points(cfg.max_points);std::vector<Detection> detections(cfg.max_detections);
  size_t comparisons=0,mismatches=0,measurement_comparisons=0,frames=0,max_projected=0;
  faults(gpu,cfg);synthetic_parity(gpu,comparisons,mismatches);synthetic_core_parity(gpu,measurement_comparisons);
  const auto fixed_bytes=gpu.allocated_bytes();size_t initial_rss=0,max_rss=0;
  for(int rep=0;rep<repeats;++rep) {
    std::ifstream in(replay,std::ios::binary);char magic[8];
    if(!in.read(magic,8)||std::memcmp(magic,"G2FUS001",8))throw std::runtime_error("invalid G2FUS001 replay");
    uint32_t count;read(in,count);if(count>100000)throw std::runtime_error("too many replay frames");
    if(uint64_t(count)*repeats>1000000)throw std::runtime_error("timing sample capacity exceeded");
    if(rep==0) {
      const size_t capacity=size_t(count)*repeats;
      samples.cpu.reserve(capacity);samples.total.reserve(capacity);
      samples.upload.reserve(capacity);samples.kernels.reserve(capacity);samples.download.reserve(capacity);
      samples.full_cpu.reserve(capacity);samples.full_cuda.reserve(capacity);
    }
    reference.reset_tracks();accelerated.reset_tracks();
    for(uint32_t frame=0;frame<count;++frame) {
      Context ctx;ctx.tf_valid=true;ctx.output_frame=camera.frame;
      // First five frames exercise nominal fail-closed behavior. Remaining
      // synthetic trust only exercises association; it validates no calibration.
      ctx.intrinsics_trusted=ctx.extrinsics_trusted=ctx.timing_trusted=frame>=5;
      uint32_t n,b;read(in,ctx.image_ns);read(in,ctx.cloud_ns);read(in,n);
      if(n>points.size())throw std::runtime_error("replay cloud too large");
      for(auto& v:ctx.camera_from_cloud.rotation)read(in,v);
      for(auto& v:ctx.camera_from_cloud.translation)read(in,v);
      if(!in.read(reinterpret_cast<char*>(points.data()),n*sizeof(Point)))throw std::runtime_error("truncated cloud");
      read(in,b);if(b>detections.size())throw std::runtime_error("replay detections too large");
      for(uint32_t i=0;i<b;++i) {
        auto& d=detections[i];read(in,d.x0);read(in,d.y0);read(in,d.x1);read(in,d.y1);read(in,d.score);read(in,d.label_id);
      }
      const auto cpu_status=reference.process(camera,ctx,points.data(),n,detections.data(),b);
      const auto cuda_status=accelerated.process(camera,ctx,points.data(),n,detections.data(),b);
      if(cpu_status!=cuda_status)throw std::runtime_error("full core frame status parity failed");
      if(reference.projected().size()!=accelerated.projected().size())throw std::runtime_error("full core projected count mismatch");
      for(size_t i=0;i<reference.projected().size();++i) {
        ++comparisons;if(reference.projected()[i].visible!=accelerated.projected()[i].visible)++mismatches;
      }
      measurement_parity(reference,accelerated,measurement_comparisons);
      if(frame>=5) {
        // Rejected frames do not invoke the GPU; never reuse prior event timing.
        const auto stats=reference.timing().visibility_ms>0?gpu.stats():CudaVisibility::Stats{};
        samples.add(reference.timing().visibility_ms,stats);
        samples.full_cpu.push_back(reference.timing().total_ms);samples.full_cuda.push_back(accelerated.timing().total_ms);
      }
      if(frame==5) {const auto rss=rss_kib();if(!initial_rss)initial_rss=rss;max_rss=std::max(max_rss,rss);}
      if(gpu.allocated_bytes()!=fixed_bytes)throw std::runtime_error("CUDA storage grew");
      ++frames;max_projected=std::max(max_projected,reference.projected().size());
    }
    char extra;if(in.read(&extra,1))throw std::runtime_error("trailing replay bytes");
  }
  // Exercise repeated GPU execution without allocating any per-call storage.
  std::vector<Projected> steady_points(128);
  for(size_t i=0;i<steady_points.size();++i) {
    steady_points[i].pixel=uint32_t(i*73);steady_points[i].camera_point={0,0,2};
  }
  for(int i=0;i<100;++i)if(!gpu.run(steady_points,camera.width,camera.height))throw std::runtime_error(gpu.error());
  const auto sustained_initial_rss=rss_kib();
  for(int i=0;i<20000;++i) {
    if(!gpu.run(steady_points,camera.width,camera.height))throw std::runtime_error(gpu.error());
    if(gpu.allocated_bytes()!=fixed_bytes)throw std::runtime_error("sustained CUDA storage grew");
  }
  const auto sustained_final_rss=rss_kib();
  const auto cpu_p95=Samples::quantile(samples.cpu,.95),cuda_p95=Samples::quantile(samples.total,.95);
  std::cout<<"{\"backend\":\"cuda_visibility\",\"transfer\":\""
    <<(mode==CudaVisibility::TransferMode::ExplicitCopy?"explicit_copy":"mapped_pinned")
    <<"\",\"frames\":"<<frames<<",\"samples\":"<<samples.total.size()<<",\"max_projected\":"<<max_projected
    <<",\"boolean_comparisons\":"<<comparisons<<",\"mismatches\":"<<mismatches<<",\"fault_checks_passed\":true"
    <<",\"exact_object_measurement_comparisons\":"<<measurement_comparisons<<",\"synthetic_trust_for_association_only\":true"
    <<",\"allocated_bytes\":"<<fixed_bytes<<",\"initial_rss_kib\":"<<initial_rss<<",\"final_rss_kib\":"<<rss_kib()
    <<",\"sustained_gpu_calls\":20000,\"sustained_initial_rss_kib\":"<<sustained_initial_rss
    <<",\"sustained_final_rss_kib\":"<<sustained_final_rss
    <<",\"sampled_max_rss_kib\":"<<max_rss<<",\"p95_speedup\":"<<(cuda_p95>0?cpu_p95/cuda_p95:0)
    <<",\"meets_20_percent_improvement\":"<<(cpu_p95>0&&cuda_p95<=.8*cpu_p95?"true":"false")
    <<",\"metric_accuracy_validated\":false,\"cpu_visibility\":";
  Samples::print(samples.cpu);std::cout<<",\"cuda_total_including_staging_and_sync\":";Samples::print(samples.total);
  std::cout<<",\"cuda_upload\":";Samples::print(samples.upload);std::cout<<",\"cuda_kernels\":";Samples::print(samples.kernels);
  std::cout<<",\"cuda_download\":";Samples::print(samples.download);
  std::cout<<",\"cpu_full_fusion\":";Samples::print(samples.full_cpu);
  std::cout<<",\"cuda_full_fusion\":";Samples::print(samples.full_cuda);std::cout<<"}\n";
  if(mismatches)throw std::runtime_error("CUDA visibility differs from CPU reference");
}
}
int main(int argc,char** argv) try {
  if(argc<2) {std::cerr<<"Usage: cuda_visibility_benchmark replay.g2f [repeats=1] [explicit|mapped|both]\n";return 2;}
  const int repeats=argc>2?std::stoi(argv[2]):1;
  if(repeats<1||repeats>1000)throw std::runtime_error("repeats outside [1,1000]");
  const std::string mode=argc>3?argv[3]:"both";
  if(mode!="explicit"&&mode!="mapped"&&mode!="both")throw std::runtime_error("unknown transfer mode");
  if(mode=="explicit"||mode=="both")benchmark(argv[1],repeats,CudaVisibility::TransferMode::ExplicitCopy);
  if(mode=="mapped"||mode=="both")benchmark(argv[1],repeats,CudaVisibility::TransferMode::MappedPinned);
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
