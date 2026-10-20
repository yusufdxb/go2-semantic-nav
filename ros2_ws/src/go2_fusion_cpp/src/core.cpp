#include "go2_fusion_cpp/core.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <numeric>

namespace go2_fusion {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
bool finite(Point p) { return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z); }
double distance(Point a,Point b) { return std::hypot(std::hypot(a.x-b.x,a.y-b.y),a.z-b.z); }
bool contains(const Detection& d,const Projected& p,uint32_t width) {
  if(p.u<d.x0||p.u>=d.x1||p.v<d.y0||p.v>=d.y1)return false;
  if(!d.mask)return true;
  uint32_t x=p.pixel%width,y=p.pixel/width;
  if(!d.mask_width)return d.mask[p.pixel]!=0;
  if(x<d.mask_x||y<d.mask_y||x>=uint64_t(d.mask_x)+d.mask_width||y>=uint64_t(d.mask_y)+d.mask_height)return false;
  return d.mask[size_t(y-d.mask_y)*d.mask_width+x-d.mask_x]!=0;
}
}
Point Transform::apply(Point p) const {
  return {float(rotation[0]*p.x+rotation[1]*p.y+rotation[2]*p.z+translation[0]),
    float(rotation[3]*p.x+rotation[4]*p.y+rotation[5]*p.z+translation[1]),
    float(rotation[6]*p.x+rotation[7]*p.y+rotation[8]*p.z+translation[2])};
}
bool Transform::valid() const {
  for(double v:rotation) if(!std::isfinite(v)) return false;
  for(double v:translation) if(!std::isfinite(v)) return false;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    double dot=0; for(int k=0;k<3;++k) dot+=rotation[i*3+k]*rotation[j*3+k];
    if(std::abs(dot-(i==j?1:0))>1e-5) return false;
  }
  const auto& r=rotation;
  double det=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
  return std::abs(det-1)<1e-5;
}
bool Camera::valid() const {
  if(!width||!height||frame.empty()||!std::isfinite(fx)||!std::isfinite(fy)||fx<=0||fy<=0||
    !std::isfinite(cx)||!std::isfinite(cy)||cx<0||cy<0||cx>=width||cy>=height) return false;
  for(double d:distortion) if(!std::isfinite(d)) return false;
  return true;
}
const char* status_name(Status s) {
  switch(s) {
#define S(x) case Status::x:return #x
    S(Valid);S(UntrustedCalibration);S(MissingTransform);S(TimestampSkew);S(MalformedInput);
    S(CapacityExceeded);S(InsufficientSupport);S(AmbiguousDepth);S(OutOfOrder);S(BackendError);
#undef S
  } return "Unknown";
}
Engine::Engine(Config cfg,VisibilityFunction accelerator):cfg_(cfg),accelerator_(std::move(accelerator)) {
  if(!cfg.max_points||cfg.max_points>10000000||!cfg.max_detections||cfg.max_detections>1024||
    !cfg.max_tracks||cfg.max_tracks>4096||!cfg.max_width||!cfg.max_height||
    uint64_t(cfg.max_width)*cfg.max_height>16000000||cfg.occlusion_radius_px<0||cfg.occlusion_radius_px>128||
    cfg.min_support<2||cfg.max_skew_ns<0||cfg.track_ttl_ns<=0) throw std::invalid_argument("invalid fusion capacity/configuration");
  const double values[]={cfg.occlusion_margin_m,cfg.min_depth_m,cfg.max_depth_m,cfg.max_normalized_radius,
    cfg.cluster_gap_m,cfg.min_cluster_fraction,cfg.lidar_sigma_m,cfg.calibration_sigma_m,cfg.max_object_speed_mps,cfg.track_gate_m};
  for(double v:values) if(!std::isfinite(v)||v<0) throw std::invalid_argument("nonfinite/negative configuration");
  for(double v:{cfg.lidar_sigma_m,cfg.calibration_sigma_m,cfg.max_object_speed_mps})
    if(!std::isfinite(v*v)) throw std::invalid_argument("uncertainty configuration overflows");
  if(cfg.min_depth_m<=0||cfg.max_depth_m<=cfg.min_depth_m||cfg.max_normalized_radius<=0||
    cfg.cluster_gap_m<=0||cfg.min_cluster_fraction<=0.5||cfg.min_cluster_fraction>1||cfg.track_gate_m<=0)
    throw std::invalid_argument("invalid fusion thresholds");
  size_t pixels=size_t(cfg.max_width)*cfg.max_height;
  if(!accelerator_) {
    depth_.resize(pixels);scratch_.resize(pixels);
    deque_.resize(std::max(cfg.max_width,cfg.max_height));
  }
  projected_.reserve(cfg.max_points);candidates_.reserve(cfg.max_points);
  results_.reserve(cfg.max_detections);tracks_.reserve(cfg.max_tracks);
}
size_t Engine::working_bytes() const {
  return (depth_.capacity()+scratch_.capacity())*sizeof(float)+
    deque_.capacity()*sizeof(int)+projected_.capacity()*sizeof(Projected)+
    candidates_.capacity()*sizeof(uint32_t)+results_.capacity()*sizeof(Measurement)+tracks_.capacity()*sizeof(Track);
}
void Engine::reset_tracks() { tracks_.clear();last_stamp_=0;last_frame_.clear(); }
void Engine::visibility(uint32_t w,uint32_t h) {
  const uint32_t diameter=2*cfg_.occlusion_radius_px+1;
  // Sparse LiDAR projections need only their local visibility footprints.
  // Bound this path by one image worth of probes; dense inputs use the
  // radius-independent separable filter below.
  if(uint64_t(projected_.size())*std::min(w,diameter)*std::min(h,diameter)<=uint64_t(w)*h) {
    for(auto& p:projected_) {
      p.visible=p.camera_point.z==depth_[p.pixel];
      if(!p.visible)continue;
      const int x=p.pixel%w,y=p.pixel/w,r=cfg_.occlusion_radius_px;
      const int x0=std::max(0,x-r),x1=std::min(int(w)-1,x+r);
      const int y0=std::max(0,y-r),y1=std::min(int(h)-1,y+r);
      for(int v=y0;v<=y1&&p.visible;++v)
        for(int u=x0;u<=x1;++u)
          if(p.camera_point.z>depth_[size_t(v)*w+u]+cfg_.occlusion_margin_m) {
            p.visible=false;break;
          }
    }
    return;
  }
  // Two separable sliding minima, O(width*height) independent of window radius.
  auto pass=[&](const std::vector<float>& in,int lines,int length,int stride,int step,auto emit) {
    for(int line=0;line<lines;++line) {
      int head=0,tail=0,added=-1,base=line*step;
      for(int i=0;i<length;++i) {
        int end=std::min(length-1,i+cfg_.occlusion_radius_px);
        while(added<end) {
          ++added;
          while(tail>head&&in[base+deque_[tail-1]*stride]>=in[base+added*stride]) --tail;
          deque_[tail++]=added;
        }
        while(head<tail&&deque_[head]<i-cfg_.occlusion_radius_px) ++head;
        emit(base+i*stride,in[base+deque_[head]*stride]);
      }
    }
  };
  pass(depth_,h,w,1,w,[&](size_t pixel,float nearest){scratch_[pixel]=nearest;});
  // The horizontal minima no longer depend on depth_. Remove occluded raster
  // winners in place while computing vertical minima, avoiding a third image.
  pass(scratch_,w,h,w,1,[&](size_t pixel,float nearest){
    if(depth_[pixel]>nearest+cfg_.occlusion_margin_m)
      depth_[pixel]=std::numeric_limits<float>::infinity();
  });
  // Only nearest-depth returns survive. Association counts each pixel once.
  for(auto& p:projected_) p.visible=p.camera_point.z==depth_[p.pixel];
}
Measurement Engine::associate(const Detection& d,size_t index,const Camera& cam,const Context& ctx) {
  Measurement m;m.label_id=d.label_id;m.stamp_ns=ctx.image_ns;
  candidates_.clear();
  for(uint32_t i=0;i<projected_.size();++i) {
    const auto& p=projected_[i];
    if(!p.visible||p.owner!=int32_t(index)) continue;
    candidates_.push_back(i);
  }
  if(candidates_.size()<cfg_.min_support) return m;
  std::sort(candidates_.begin(),candidates_.end(),[&](uint32_t a,uint32_t b){
    const auto& p=projected_[a];const auto& q=projected_[b];
    return p.camera_point.z==q.camera_point.z?p.pixel<q.pixel:p.camera_point.z<q.camera_point.z;
  });
  // Duplicated returns at the same raster cell must count once.
  candidates_.erase(std::unique(candidates_.begin(),candidates_.end(),[&](uint32_t a,uint32_t b){
    return projected_[a].pixel==projected_[b].pixel;
  }),candidates_.end());
  size_t best_start=0,best_count=0,start=0;
  for(size_t i=1;i<=candidates_.size();++i) {
    if(i==candidates_.size()||projected_[candidates_[i]].camera_point.z-projected_[candidates_[i-1]].camera_point.z>cfg_.cluster_gap_m) {
      if(i-start>best_count){best_count=i-start;best_start=start;}start=i;
    }
  }
  if(best_count<cfg_.min_support) return m;
  if(double(best_count)/candidates_.size()<cfg_.min_cluster_fraction) { m.status=Status::AmbiguousDepth;return m; }
  std::array<double,3> mean{};
  for(size_t i=best_start;i<best_start+best_count;++i){auto p=projected_[candidates_[i]].camera_point;mean[0]+=p.x;mean[1]+=p.y;mean[2]+=p.z;}
  for(auto& v:mean) v/=best_count;
  std::array<double,9> cov{};
  for(size_t i=best_start;i<best_start+best_count;++i) {
    auto p=projected_[candidates_[i]].camera_point;
    double delta[]={p.x-mean[0],p.y-mean[1],p.z-mean[2]};
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)cov[a*3+b]+=delta[a]*delta[b]/(best_count-1);
  }
  // Surface spread retained conservatively; common calibration/time errors do not shrink with N.
  double skew=double(std::abs(ctx.image_ns-ctx.cloud_ns))*1e-9;
  double floor=cfg_.lidar_sigma_m*cfg_.lidar_sigma_m+cfg_.calibration_sigma_m*cfg_.calibration_sigma_m+
    std::pow(skew*cfg_.max_object_speed_mps,2);
  for(int i=0;i<3;++i)cov[i*3+i]+=floor;
  m.position=ctx.output_from_camera.apply({float(mean[0]),float(mean[1]),float(mean[2])});
  if(!finite(m.position)) {m.status=Status::MalformedInput;return m;}
  const auto& r=ctx.output_from_camera.rotation;m.covariance.fill(0);
  for(int a=0;a<3;++a)for(int b=0;b<3;++b)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    m.covariance[a*3+b]+=r[a*3+i]*cov[i*3+j]*r[b*3+j];
  for(double v:m.covariance) if(!std::isfinite(v)) {
    m.position={NAN,NAN,NAN};m.covariance.fill(NAN);m.status=Status::MalformedInput;return m;
  }
  m.support=best_count;m.status=Status::Valid;
  (void)cam;return m;
}
void Engine::track(const Context& ctx) {
  if(ctx.output_frame!=last_frame_) tracks_.clear();
  last_frame_=ctx.output_frame;
  tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const Track& t){return ctx.image_ns-t.stamp>cfg_.track_ttl_ns;}),tracks_.end());
  for(auto& t:tracks_)t.used=false;
  for(auto& m:results_) {
    if(m.status!=Status::Valid) continue;
    Track* best=nullptr;double gate=cfg_.track_gate_m;
    for(auto& t:tracks_)if(!t.used&&t.label==m.label_id) {
      double dist=distance(t.position,m.position);
      if(dist<gate){gate=dist;best=&t;}
    }
    if(best){m.track_id=best->id;best->position=m.position;best->stamp=ctx.image_ns;best->used=true;}
    else if(tracks_.size()<cfg_.max_tracks) {m.track_id=next_id_++;tracks_.push_back({m.track_id,m.label_id,m.position,ctx.image_ns,true});}
  }
  last_stamp_=ctx.image_ns;
}
Status Engine::process(const Camera& cam,const Context& ctx,const Point* points,size_t count,const Detection* detections,size_t num) {
  auto all=Clock::now();timing_={};projected_.clear();results_.clear();
  Status status=Status::Valid;
  if(count>cfg_.max_points||num>cfg_.max_detections||cam.width>cfg_.max_width||cam.height>cfg_.max_height) status=Status::CapacityExceeded;
  else if(!cam.valid()||(count&&!points)||(num&&!detections)||ctx.image_ns<=0||ctx.cloud_ns<=0||ctx.output_frame.empty())status=Status::MalformedInput;
  else if(!ctx.tf_valid||!ctx.camera_from_cloud.valid()||!ctx.output_from_camera.valid())status=Status::MissingTransform;
  else if(std::abs(ctx.image_ns-ctx.cloud_ns)>cfg_.max_skew_ns)status=Status::TimestampSkew;
  else if(last_stamp_&&ctx.image_ns<=last_stamp_)status=Status::OutOfOrder;
  for(size_t i=0;status==Status::Valid&&i<num;++i) {
    const auto& d=detections[i];
    if(!std::isfinite(d.x0)||!std::isfinite(d.x1)||!std::isfinite(d.y0)||!std::isfinite(d.y1)||!std::isfinite(d.score)||d.score<0||d.score>1||
      d.x0<0||d.y0<0||d.x1>cam.width||d.y1>cam.height||d.x1<=d.x0||d.y1<=d.y0||
      (d.mask&&((!d.mask_width&&d.mask_size!=size_t(cam.width)*cam.height)||
        (d.mask_width&&(!d.mask_height||uint64_t(d.mask_x)+d.mask_width>cam.width||uint64_t(d.mask_y)+d.mask_height>cam.height||
          d.mask_size!=size_t(d.mask_width)*d.mask_height)))))status=Status::MalformedInput;
  }
  auto fill_invalid=[&](Status s) {
    for(size_t i=0;i<std::min(num,cfg_.max_detections);++i){Measurement m;m.status=s;m.stamp_ns=ctx.image_ns;if(detections)m.label_id=detections[i].label_id;results_.push_back(m);}
  };
  if(status!=Status::Valid){fill_invalid(status);timing_.total_ms=elapsed(all);return status;}
  auto t=Clock::now();size_t pixels=size_t(cam.width)*cam.height;
  if(!accelerator_) std::fill_n(depth_.begin(),pixels,std::numeric_limits<float>::infinity());
  for(size_t i=0;i<count;++i) {
    if(!finite(points[i]))continue;
    auto p=ctx.camera_from_cloud.apply(points[i]);
    if(!finite(p)||p.z<=cfg_.min_depth_m||p.z>=cfg_.max_depth_m)continue;
    double x=p.x/p.z,y=p.y/p.z,r2=x*x+y*y;
    if(r2>cfg_.max_normalized_radius*cfg_.max_normalized_radius)continue;
    const auto& d=cam.distortion;double radial=1+d[0]*r2+d[1]*r2*r2+d[4]*r2*r2*r2;
    double u=cam.fx*(x*radial+2*d[2]*x*y+d[3]*(r2+2*x*x))+cam.cx;
    double v=cam.fy*(y*radial+d[2]*(r2+2*y*y)+2*d[3]*x*y)+cam.cy;
    if(!std::isfinite(u)||!std::isfinite(v)||u<=-0.5||v<=-0.5||u>=cam.width-0.5||v>=cam.height-0.5)continue;
    // Half-pixel ties are rounded to nearest even, matching numpy.rint.
    uint32_t pixel=uint32_t(std::nearbyint(v))*cam.width+uint32_t(std::nearbyint(u));
    if(!accelerator_) depth_[pixel]=std::min(depth_[pixel],p.z);
    projected_.push_back({p,float(u),float(v),pixel,false});
  }
  timing_.projection_ms=elapsed(t);t=Clock::now();
  if(accelerator_) {
    bool ok=false;
    try {ok=accelerator_(projected_,cam.width,cam.height);}catch(const std::exception&) {ok=false;}
    if(!ok) {
      projected_.clear();reset_tracks();fill_invalid(Status::BackendError);
      timing_.visibility_ms=elapsed(t);timing_.total_ms=elapsed(all);return Status::BackendError;
    }
  } else visibility(cam.width,cam.height);
  timing_.visibility_ms=elapsed(t);t=Clock::now();
  // Diagnostic projections remain available, but nominal geometry never produces a position.
  if(!ctx.intrinsics_trusted||!ctx.extrinsics_trusted||!ctx.timing_trusted) {status=Status::UntrustedCalibration;fill_invalid(status);}
  else {
    // Exclusive point ownership prevents overlapping boxes from duplicating an object.
    // Near-equal competing scores are ambiguous and contribute to neither object.
    for(auto& p:projected_) if(p.visible) {
      int best=-1;float top=-1,second=-1;
      for(size_t i=0;i<num;++i) if(contains(detections[i],p,cam.width)) {
        float score=detections[i].score;
        if(score>top){second=top;top=score;best=i;}else second=std::max(second,score);
      }
      p.owner=(second>=0&&top-second<0.05f)?-1:best;
    }
    for(size_t i=0;i<num;++i) results_.push_back(associate(detections[i],i,cam,ctx));
    track(ctx);
  }
  timing_.association_ms=elapsed(t);timing_.total_ms=elapsed(all);return status;
}
} // namespace go2_fusion
