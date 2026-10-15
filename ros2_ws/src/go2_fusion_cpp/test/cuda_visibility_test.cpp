#include "go2_fusion_cpp/cuda_visibility.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace go2_fusion;

namespace {
void compare(CudaVisibility& gpu, const std::vector<Projected>& points,
             uint32_t w, uint32_t h, int radius, double margin, size_t& comparisons) {
  const float inf=std::numeric_limits<float>::infinity();
  std::vector<float> raster(size_t(w)*h,inf),horizontal(raster.size(),inf);
  for(const auto& p:points)raster[p.pixel]=std::min(raster[p.pixel],p.camera_point.z);
  // Independent scalar reference, including clipped image borders.
  for(uint32_t y=0;y<h;++y)for(uint32_t x=0;x<w;++x)
    for(int col=std::max(0,int(x)-radius);col<=std::min(int(w)-1,int(x)+radius);++col)
      horizontal[size_t(y)*w+x]=std::min(horizontal[size_t(y)*w+x],raster[size_t(y)*w+col]);
  if(!gpu.run(points,w,h))throw std::runtime_error(gpu.error());
  if(gpu.size()!=points.size())throw std::runtime_error("flag count mismatch");
  for(size_t i=0;i<points.size();++i) {
    const auto& p=points[i];const int x=p.pixel%w,y=p.pixel/w;
    float nearest=inf;
    for(int row=std::max(0,y-radius);row<=std::min(int(h)-1,y+radius);++row)
      nearest=std::min(nearest,horizontal[size_t(row)*w+x]);
    const bool expected=p.camera_point.z==raster[p.pixel]&&double(p.camera_point.z)<=double(nearest)+margin;
    if(bool(gpu.flags()[i])!=expected)throw std::runtime_error("independent CUDA visibility mismatch");
    ++comparisons;
  }
}
}

int main() try {
  std::mt19937 random(5831);size_t comparisons=0;
  for(auto mode:{CudaVisibility::TransferMode::ExplicitCopy,CudaVisibility::TransferMode::MappedPinned})
    for(int radius:{0,1,2,10,16,31,64,128})for(double margin:{0.,0.3}) {
      Config cfg;cfg.occlusion_radius_px=radius;cfg.occlusion_margin_m=margin;
      CudaVisibility gpu(cfg,mode);
      for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{{1,1},{1,35},{47,1},{37,29},{65,49},{131,97}})
        for(size_t count:{size_t(0),size_t(1),size_t(17),size_t(2000)}) {
          std::vector<Projected> points(count);
          for(size_t i=0;i<count;++i) {
            points[i].pixel=random()%(shape.first*shape.second);
            points[i].camera_point.z=0.5F+float(random()%200)*0.03125F;
            if(i&&i%7==0)points[i]=points[i-1];
          }
          compare(gpu,points,shape.first,shape.second,radius,margin,comparisons);
        }
    }
  // Exercise both dispatch paths at the deployed image dimensions, with ties
  // and depth differences exactly at and just beyond the margin.
  for(auto mode:{CudaVisibility::TransferMode::ExplicitCopy,CudaVisibility::TransferMode::MappedPinned}) {
    Config cfg;CudaVisibility gpu(cfg,mode);
    for(size_t count:{size_t(800),size_t(50000)}) {
      std::vector<Projected> points(count);
      for(size_t i=0;i<count;++i) {
        points[i].pixel=random()%(1280*720);
        points[i].camera_point.z=1.F+float(random()%128)*0.03125F;
      }
      points[0].pixel=0;points[0].camera_point.z=1.F;
      points[1].pixel=1;points[1].camera_point.z=1.3F;
      points[2].pixel=2;points[2].camera_point.z=std::nextafter(1.3F,0.F);
      compare(gpu,points,1280,720,10,0.3,comparisons);
      const auto bytes=gpu.allocated_bytes();
      if(gpu.run(points,1280,720,true))throw std::runtime_error("injected failure accepted");
      for(size_t i=0;i<gpu.size();++i)if(gpu.flags()[i])throw std::runtime_error("failure retained visible flag");
      compare(gpu,points,1280,720,10,0.3,comparisons);
      if(gpu.allocated_bytes()!=bytes)throw std::runtime_error("storage grew");
    }
  }
  std::cout<<"CUDA independent visibility checks passed: "<<comparisons<<" flags\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
