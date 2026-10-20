#include "go2_fusion_cpp/core.hpp"
#ifdef GO2_FUSION_HAS_CUDA
#include "go2_fusion_cpp/cuda_visibility.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/resource.h>
using namespace go2_fusion;
namespace {
template<class T> void read(std::istream& in,T& value) {
  if(!in.read(reinterpret_cast<char*>(&value),sizeof(value)))throw std::runtime_error("truncated replay input");
}
size_t rss_kib() {
  std::ifstream in("/proc/self/status");std::string line;
  while(std::getline(in,line))if(line.rfind("VmRSS:",0)==0)return std::stoull(line.substr(6));
  return 0;
}
struct Samples {
  std::vector<double> total,projection,visibility,association;
  void add(const Timing& t){total.push_back(t.total_ms);projection.push_back(t.projection_ms);visibility.push_back(t.visibility_ms);association.push_back(t.association_ms);}
  static void print(std::vector<double> v) {
    std::sort(v.begin(),v.end());
    auto q=[&](double p){return v.empty()?0:v[size_t(std::ceil(p*v.size()))-1];};
    std::cout<<"{\"p50_ms\":"<<q(.5)<<",\"p95_ms\":"<<q(.95)<<",\"p99_ms\":"<<q(.99)<<",\"max_ms\":"<<(v.empty()?0:v.back())<<"}";
  }
};
}
int main(int argc,char** argv) try {
  if(argc<2){std::cerr<<"Usage: fusion_benchmark replay.bin [repeats=1] [exercise_association=0] [projected.csv|-] [cpu|cuda|cuda_mapped]\n";return 2;}
  int repeats=argc>2?std::stoi(argv[2]):1;
  bool exercise=argc>3&&std::stoi(argv[3])!=0;
  if(repeats<1||repeats>1000)throw std::runtime_error("repeats outside [1,1000]");
  const std::string backend=argc>5?argv[5]:"cpu";
  Config cfg;Engine::VisibilityFunction accelerator;
  size_t cuda_bytes=0;
#ifdef GO2_FUSION_HAS_CUDA
  std::shared_ptr<CudaVisibility> gpu;
  if(backend=="cuda"||backend=="cuda_mapped") {
    gpu=std::make_shared<CudaVisibility>(cfg,backend=="cuda"?CudaVisibility::TransferMode::ExplicitCopy:CudaVisibility::TransferMode::MappedPinned);
    cuda_bytes=gpu->allocated_bytes();
    accelerator=[gpu](std::vector<Projected>& points,uint32_t width,uint32_t height) {
      if(!gpu->run(points,width,height)||gpu->size()!=points.size())return false;
      for(size_t i=0;i<points.size();++i)points[i].visible=gpu->flags()[i]!=0;
      return true;
    };
  } else if(backend!="cpu")throw std::runtime_error("unknown backend");
#else
  if(backend!="cpu")throw std::runtime_error("CUDA backend not enabled at build time");
#endif
  Camera camera;Engine engine(cfg,accelerator);Samples samples;
  std::vector<Point> points(100000);std::vector<Detection> detections(64);
  size_t frames=0,invalid=0,valid_measurements=0,max_points=0,initial_rss=0,max_rss=0;
  for(int rep=0;rep<repeats;++rep){
    engine.reset_tracks();std::ifstream in(argv[1],std::ios::binary);
    char magic[8];if(!in.read(magic,8)||std::memcmp(magic,"G2FUS001",8))throw std::runtime_error("wrong replay magic");
    uint32_t count;read(in,count);if(count>100000)throw std::runtime_error("too many frames");
    if(uint64_t(count)*repeats>1000000)throw std::runtime_error("timing sample capacity exceeded");
    if(rep==0) {
      size_t capacity=size_t(count)*repeats;
      samples.total.reserve(capacity);samples.projection.reserve(capacity);
      samples.visibility.reserve(capacity);samples.association.reserve(capacity);
    }
    for(uint32_t f=0;f<count;++f){
      Context ctx;ctx.output_frame=camera.frame;ctx.tf_valid=true;
      ctx.intrinsics_trusted=ctx.extrinsics_trusted=ctx.timing_trusted=exercise;
      uint32_t n,b;read(in,ctx.image_ns);read(in,ctx.cloud_ns);read(in,n);
      if(n>points.size())throw std::runtime_error("cloud capacity exceeded");
      for(auto& v:ctx.camera_from_cloud.rotation)read(in,v);
      for(auto& v:ctx.camera_from_cloud.translation)read(in,v);
      if(!in.read(reinterpret_cast<char*>(points.data()),n*sizeof(Point)))throw std::runtime_error("truncated points");
      read(in,b);if(b>detections.size())throw std::runtime_error("detection capacity exceeded");
      for(uint32_t i=0;i<b;++i){auto& d=detections[i];read(in,d.x0);read(in,d.y0);read(in,d.x1);read(in,d.y1);read(in,d.score);read(in,d.label_id);}
      auto status=engine.process(camera,ctx,points.data(),n,detections.data(),b);
      if(status!=Status::Valid)++invalid;
      for(const auto& m:engine.measurements())if(m.status==Status::Valid)++valid_measurements;
      // Warm each fresh replay for five frames. Remaining bounded timings are retained.
      if(f>=5)samples.add(engine.timing());
      if(f==5){size_t rss=rss_kib();if(!initial_rss)initial_rss=rss;max_rss=std::max(max_rss,rss);}
      max_points=std::max(max_points,size_t(n));++frames;
      if(argc>4&&std::string(argv[4])!="-"&&rep==0&&f==5){std::ofstream out(argv[4]);out<<"u,v,z,visible\n";for(const auto& p:engine.projected())out<<p.u<<','<<p.v<<','<<p.camera_point.z<<','<<p.visible<<'\n';}
    }
    char extra;if(in.read(&extra,1))throw std::runtime_error("unexpected trailing replay data");
  }
  rusage usage{};getrusage(RUSAGE_SELF,&usage);
  std::cout<<"{\"backend\":\""<<backend<<"\",\"cuda_working_bytes\":"<<cuda_bytes<<",\"frames\":"<<frames<<",\"samples\":"<<samples.total.size()<<",\"max_points\":"<<max_points
    <<",\"invalid_frames\":"<<invalid<<",\"valid_measurements\":"<<valid_measurements<<",\"association_exercised_with_synthetic_trust\":"<<(exercise?"true":"false")
    <<",\"metric_accuracy_validated\":false,\"sensor_to_output_ms\":null,\"sensor_to_output_reason\":\"Exposure clock and detector/output timestamps not calibrated in recording\",\"core_working_bytes\":"<<engine.working_bytes()<<",\"initial_rss_kib\":"<<initial_rss<<",\"final_rss_kib\":"<<rss_kib()<<",\"sampled_max_rss_kib\":"<<max_rss<<",\"process_high_water_rss_kib\":"<<usage.ru_maxrss
    <<",\"fusion\":";Samples::print(samples.total);std::cout<<",\"projection\":";Samples::print(samples.projection);std::cout<<",\"visibility\":";Samples::print(samples.visibility);std::cout<<",\"association\":";Samples::print(samples.association);std::cout<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
