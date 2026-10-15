#include "go2_fusion_cpp/core.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace go2_fusion;
namespace {
void require(bool condition, const std::string& reason) {
  if (!condition) throw std::runtime_error(reason);
}
void near(double actual, double expected, double tolerance=1e-6) {
  require(std::isfinite(actual) && std::abs(actual-expected)<=tolerance,
          "expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
Config configuration() {
  Config c;
  c.max_width=128; c.max_height=96; c.max_points=256;
  c.max_detections=8; c.max_tracks=4; c.min_support=3;
  c.occlusion_radius_px=2;
  return c;
}
Camera camera() {
  Camera c; c.width=64; c.height=48; c.fx=40; c.fy=40; c.cx=32; c.cy=24;
  return c;
}
Context context() {
  Context c; c.image_ns=1791936000000000000LL; c.cloud_ns=c.image_ns;
  c.intrinsics_trusted=true; c.extrinsics_trusted=true; c.timing_trusted=true; c.tf_valid=true;
  return c;
}
Detection detection() {
  Detection d; d.x1=64; d.y1=48; d.score=0.9F; d.label_id=7; return d;
}
Point at_pixel(double u, double v, double z) {
  return {float((u-32)*z/40),float((v-24)*z/40),float(z)};
}
std::vector<Point> surface() {
  return {at_pixel(20,16,2),at_pixel(28,24,2),at_pixel(36,32,2)};
}
Status process(Engine& engine, const Context& c, const std::vector<Point>& p,
               const std::vector<Detection>& d={detection()}, const Camera& cam=camera()) {
  return engine.process(cam,c,p.data(),p.size(),d.data(),d.size());
}
const Measurement& single(const Engine& engine) {
  require(engine.measurements().size()==1,"one measurement required");
  return engine.measurements().front();
}
void invalid(const Measurement& m, Status expected) {
  require(m.status==expected,std::string("invalid status expected ")+status_name(expected)+", got "+status_name(m.status));
  require(m.support==0 && m.track_id==0,"invalid result must not retain support or track");
  require(std::isnan(m.position.x)&&std::isnan(m.position.y)&&std::isnan(m.position.z),"invalid position must be NaN");
  for(double x:m.covariance) require(std::isnan(x),"invalid covariance must be NaN");
}
void projection_and_distortion() {
  Engine engine(configuration()); auto c=context(); auto cam=camera();
  cam.distortion={0.1,-0.02,0.003,-0.004,0.001};
  process(engine,c,{{0.4F,0.2F,2.0F}},{},cam);
  require(engine.projected().size()==1,"Brown distortion projection count");
  const auto& p=engine.projected().front();
  // Analytic Brown-Conrady evaluation at x=0.2, y=0.1, r^2=0.05.
  near(p.u,40.023601,4e-6); near(p.v,28.0218005,4e-6);
  require(p.pixel==28*64+40,"distorted pixel nearest rounding");
  c.image_ns++; c.cloud_ns++;
  process(engine,c,{{0,0,-1},{NAN,0,2},{INFINITY,0,2},{0,0,0.1F},{0,0,9},{30,0,2}},{},cam);
  require(engine.projected().empty(),"nonfinite, behind-camera, range and FOV filtering");
}
void transform_contracts() {
  Transform t; t.rotation={0,-1,0,0,0,-1,1,0,0}; t.translation={0.1,0.2,0.3};
  require(t.valid(),"proper rigid transform accepted");
  const auto q=t.apply({2,-0.3F,-0.4F}); near(q.x,0.4); near(q.y,0.6); near(q.z,2.3);
  t.rotation={-1,0,0,0,1,0,0,0,1}; require(!t.valid(),"reflection rejected");
  t=Transform{}; t.rotation[0]=1.1; require(!t.valid(),"scale rejected");
  t=Transform{}; t.translation[1]=NAN; require(!t.valid(),"nonfinite transform rejected");
  auto cam=camera(); cam.fx=0; require(!cam.valid(),"zero focal length rejected");
  cam=camera(); cam.distortion[2]=INFINITY; require(!cam.valid(),"nonfinite distortion rejected");
  cam=camera(); cam.cx=cam.width; require(!cam.valid(),"out-of-image principal point rejected");
}
void covariance_and_output_frame() {
  Engine engine(configuration()); auto c=context();
  c.output_from_camera.rotation={0,-1,0,1,0,0,0,0,1};
  c.output_from_camera.translation={1,2,3};
  require(process(engine,c,{{-0.6F,-0.4F,2},{-0.2F,0,2},{0.2F,0.4F,2}})==Status::Valid,"valid process");
  const auto m=single(engine); require(m.status==Status::Valid && m.support==3 && m.track_id!=0,"surface valid with support");
  require(m.stamp_ns==c.image_ns,"measurement source stamp");
  near(m.position.x,1); near(m.position.y,1.8); near(m.position.z,5);
  near(m.covariance[0],0.1634); near(m.covariance[4],0.1634); near(m.covariance[8],0.0034);
  near(m.covariance[1],-0.16); near(m.covariance[3],-0.16);
  // Positive semidefiniteness for independent probe directions.
  for(const auto& v:std::vector<std::array<double,3>>{{1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,-1,1}}) {
    double quadratic=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) quadratic+=v[i]*m.covariance[i*3+j]*v[j];
    require(quadratic>=0,"covariance positive semidefinite");
  }
  c.image_ns+=100000000; c.cloud_ns=c.image_ns-10000000;
  process(engine,c,{{-0.6F,-0.4F,2},{-0.2F,0,2},{0.2F,0.4F,2}});
  near(single(engine).covariance[8],0.0038,1e-8);
}
void support_deduplication() {
  Engine engine(configuration()); auto c=context(); auto p=surface();
  auto repeated=p; repeated.insert(repeated.end(),p.begin(),p.end());
  process(engine,c,repeated);
  require(single(engine).status==Status::Valid && single(engine).support==3,"duplicate returns must count once");
  c.image_ns++; c.cloud_ns++;
  process(engine,c,std::vector<Point>(20,p.front()));
  invalid(single(engine),Status::InsufficientSupport);
}
void overlapping_objects() {
  Engine engine(configuration());auto c=context();auto first=detection(),second=detection();
  second.label_id=8;
  process(engine,c,surface(),{first,second});
  require(engine.measurements().size()==2,"overlapping detections preserved as invalid outputs");
  for(const auto& m:engine.measurements())invalid(m,Status::InsufficientSupport);
  c.image_ns++;c.cloud_ns++;first.score=0.95F;second.score=0.7F;
  process(engine,c,surface(),{first,second});
  require(engine.measurements()[0].status==Status::Valid && engine.measurements()[0].support==3,
          "confident exclusive ownership gets support");
  invalid(engine.measurements()[1],Status::InsufficientSupport);
}
void accelerator_failure() {
  for(int mode=0;mode<2;++mode) {
    Engine engine(configuration(),[mode](std::vector<Projected>& points,uint32_t,uint32_t) {
      for(auto& p:points)p.visible=true;
      if(mode==1)throw std::runtime_error("backend exception");
      return false;
    });
    require(process(engine,context(),surface())==Status::BackendError,"backend failure must fail closed");
    invalid(single(engine),Status::BackendError);
    require(engine.projected().empty(),"failed backend leaves no valid projected visibility");
  }
}
void occlusion_and_ambiguity() {
  Engine engine(configuration()); auto c=context(); auto p=surface();
  for(auto q:surface()) { q.x*=2; q.y*=2; q.z=4; p.push_back(q); }
  process(engine,c,p);
  require(single(engine).status==Status::Valid && single(engine).support==3,"background on same rays occluded");
  near(single(engine).position.z,2);
  require(std::count_if(engine.projected().begin(),engine.projected().end(),[](const Projected& q){return q.visible;})==3,"three foreground winners");
  c.image_ns++; c.cloud_ns++; p=surface();
  p.push_back(at_pixel(21,17,4)); p.push_back(at_pixel(29,25,4)); p.push_back(at_pixel(37,33,4));
  process(engine,c,p); near(single(engine).position.z,2);
  require(single(engine).support==3,"nearby background rejected by visibility footprint");
  c.image_ns++; c.cloud_ns++;
  process(engine,c,{at_pixel(10,10,2),at_pixel(14,10,2),at_pixel(18,10,2),
                    at_pixel(40,35,4),at_pixel(44,35,4),at_pixel(48,35,4)});
  invalid(single(engine),Status::AmbiguousDepth);
}
void masks_and_malformed_masks() {
  Engine engine(configuration()); auto c=context(); auto p=surface(); auto d=detection();
  std::vector<uint8_t> mask(64*48,0);
  for(const auto& q:p) { const auto u=std::lround(q.x/q.z*40+32); const auto v=std::lround(q.y/q.z*40+24); mask[v*64+u]=1; }
  d.mask=mask.data(); d.mask_size=mask.size();
  process(engine,c,p,{d}); require(single(engine).status==Status::Valid,"full-image mask support");
  c.image_ns++; c.cloud_ns++; mask.assign(mask.size(),0);
  process(engine,c,p,{d}); invalid(single(engine),Status::InsufficientSupport);
  c.image_ns++; c.cloud_ns++; d.mask_size--;
  require(process(engine,c,p,{d})==Status::MalformedInput,"short mask rejected"); invalid(single(engine),Status::MalformedInput);
  // Mask ROI origin must be applied before indexing, never as full-image stride.
  c.image_ns++; c.cloud_ns++; std::vector<uint8_t> roi(24*24,0);
  d.mask=roi.data(); d.mask_x=16; d.mask_y=12; d.mask_width=24; d.mask_height=24; d.mask_size=roi.size();
  roi[(16-12)*24+20-16]=1; roi[(24-12)*24+28-16]=1; roi[(32-12)*24+36-16]=1;
  process(engine,c,p,{d}); require(single(engine).status==Status::Valid && single(engine).support==3,"ROI mask geometry");
  c.image_ns++; c.cloud_ns++; d.mask_x=std::numeric_limits<uint32_t>::max();
  require(process(engine,c,p,{d})==Status::MalformedInput,"overflowing ROI origin rejected");
  c.image_ns++; c.cloud_ns++; d.mask_x=16; d.mask_height=0;
  require(process(engine,c,p,{d})==Status::MalformedInput,"empty ROI height rejected");
}
void visibility_reference_parity() {
  std::mt19937 random(76421);
  const std::vector<std::pair<uint32_t,uint32_t>> shapes={
    {1,1},{1,13},{17,1},{7,11},{31,17},{64,48}};
  for(int radius=0;radius<=128;++radius) {
    auto cfg=configuration();cfg.occlusion_radius_px=radius;
    cfg.occlusion_margin_m=radius%2?0:0.3;
    Engine engine(cfg);auto ctx=context();
    for(const auto& shape:shapes) {
      Camera cam;cam.width=shape.first;cam.height=shape.second;
      cam.fx=cam.width;cam.fy=cam.height;
      cam.cx=(cam.width-1)*0.5;cam.cy=(cam.height-1)*0.5;
      std::vector<Point> points;
      for(size_t i=0;i<100;++i) {
        const auto u=random()%cam.width,v=random()%cam.height;
        const float z=1+float(random()%48)*0.125F;
        points.push_back({float((u-cam.cx)*z/cam.fx),float((v-cam.cy)*z/cam.fy),z});
        if(i%7==0)points.push_back(points.back());
      }
      ctx.image_ns++;ctx.cloud_ns=ctx.image_ns;
      require(process(engine,ctx,points,{},cam)==Status::Valid,"visibility reference input accepted");
      // Independent square-neighborhood reference using the unmodified
      // projected depths. No separable filter or in-place pruning is reused.
      std::vector<float> depth(size_t(cam.width)*cam.height,std::numeric_limits<float>::infinity());
      for(const auto& p:engine.projected())depth[p.pixel]=std::min(depth[p.pixel],p.camera_point.z);
      for(const auto& p:engine.projected()) {
        const int x=p.pixel%cam.width,y=p.pixel/cam.width;
        float nearest=std::numeric_limits<float>::infinity();
        for(int v=std::max(0,y-radius);v<=std::min(int(cam.height)-1,y+radius);++v)
          for(int u=std::max(0,x-radius);u<=std::min(int(cam.width)-1,x+radius);++u)
            nearest=std::min(nearest,depth[size_t(v)*cam.width+u]);
        const bool expected=p.camera_point.z==depth[p.pixel]&&p.camera_point.z<=nearest+cfg.occlusion_margin_m;
        require(p.visible==expected,"visibility parity for radius "+std::to_string(radius)+
          " at "+std::to_string(cam.width)+"x"+std::to_string(cam.height));
      }
      ctx.image_ns++;ctx.cloud_ns=ctx.image_ns;
      process(engine,ctx,{}, {},cam);
      require(engine.projected().empty(),"empty cloud following populated visibility");
    }
  }
}
void trust_and_missing_geometry() {
  for(int field=0;field<3;++field) {
    Engine engine(configuration()); auto c=context();
    if(field==0)c.intrinsics_trusted=false;
    if(field==1)c.extrinsics_trusted=false;
    if(field==2)c.timing_trusted=false;
    require(process(engine,c,surface())==Status::UntrustedCalibration,"untrusted geometry must fail closed");
    invalid(single(engine),Status::UntrustedCalibration);
    require(engine.projected().size()==3,"nominal projection remains diagnostic only");
  }
  for(int field=0;field<3;++field) {
    Engine engine(configuration()); auto c=context();
    if(field==0)c.tf_valid=false;
    if(field==1)c.camera_from_cloud.rotation[0]=2;
    if(field==2)c.output_from_camera.translation[0]=NAN;
    require(process(engine,c,surface())==Status::MissingTransform,"invalid TF must fail closed");
    invalid(single(engine),Status::MissingTransform);
    require(engine.projected().empty(),"missing transform produces no projection");
  }
}
void timestamp_limits() {
  auto c=context(); const auto p=surface();
  for(const int direction:{-1,1}) {
    for(const int64_t offset:{49999999LL,50000000LL,50000001LL}) {
      Engine engine(configuration()); c=context(); c.cloud_ns=c.image_ns+direction*offset;
      const auto expected=offset>50000000?Status::TimestampSkew:Status::Valid;
      require(process(engine,c,p)==expected,"epoch timestamp skew boundary with integer nanoseconds");
      if(expected!=Status::Valid) invalid(single(engine),expected);
    }
  }
  Engine engine(configuration()); c=context(); c.image_ns=0;
  require(process(engine,c,p)==Status::MalformedInput,"zero stamp rejected");
  c=context(); c.cloud_ns=-1;
  require(process(engine,c,p)==Status::MalformedInput,"negative stamp rejected");
}
void malformed_and_capacity() {
  Engine engine(configuration()); auto c=context(); auto d=detection();
  std::vector<Point> points(257,{0,0,2});
  require(process(engine,c,points)==Status::CapacityExceeded,"point capacity rejected"); invalid(single(engine),Status::CapacityExceeded);
  require(process(engine,c,surface(),std::vector<Detection>(9,d))==Status::CapacityExceeded,"detection capacity rejected");
  require(engine.measurements().size()==8,"invalid outputs bounded at configured capacity");
  auto cam=camera(); cam.width=129;
  require(process(engine,c,surface(),{d},cam)==Status::CapacityExceeded,"image capacity rejected");
  require(engine.process(camera(),c,nullptr,1,&d,1)==Status::MalformedInput,"null cloud rejected");
  require(engine.process(camera(),c,nullptr,0,nullptr,1)==Status::MalformedInput,"null detection array rejected");
  for(int field=0;field<4;++field) {
    d=detection();
    if(field==0)d.score=NAN;
    if(field==1)d.x0=-1;
    if(field==2)d.x1=d.x0;
    if(field==3)d.score=1.1F;
    require(process(engine,c,surface(),{d})==Status::MalformedInput,"malformed detection rejected");
    invalid(single(engine),Status::MalformedInput);
  }
}
void tracker_and_drop_behavior() {
  Engine engine(configuration()); auto c=context(); const auto p=surface();
  process(engine,c,p); const auto id=single(engine).track_id;
  require(id!=0,"first valid observation creates track");
  c.image_ns+=100000000; c.cloud_ns=c.image_ns;
  process(engine,c,p); require(single(engine).track_id==id,"matching observation preserves ID");
  auto old=c; old.image_ns--; old.cloud_ns=old.image_ns;
  require(process(engine,old,p)==Status::OutOfOrder,"old observation rejected"); invalid(single(engine),Status::OutOfOrder);
  require(process(engine,c,p)==Status::OutOfOrder,"duplicate observation rejected");
  c.image_ns+=100000000; c.cloud_ns=c.image_ns;
  process(engine,c,{}); invalid(single(engine),Status::InsufficientSupport);
  c.image_ns+=100000000; c.cloud_ns=c.image_ns;
  process(engine,c,p,{}); require(engine.measurements().empty(),"dropped detector emits no predicted position");
  c.image_ns+=100000000; c.cloud_ns=c.image_ns;
  process(engine,c,p); require(single(engine).track_id==id,"short drop preserves identity without emitting predictions");
  c.image_ns+=600000000; c.cloud_ns=c.image_ns;
  process(engine,c,p); const auto newer=single(engine).track_id;
  require(newer!=id,"expired observation gets new track");
  c.image_ns+=100000000; c.cloud_ns=c.image_ns; c.output_frame="map";
  process(engine,c,p); require(single(engine).track_id!=newer,"frame change resets track association");
  engine.reset_tracks();
  c=context(); process(engine,c,p); require(single(engine).status==Status::Valid,"explicit reset allows replay restart");
}
void sustained_capacity_stability() {
  auto cfg=configuration(); cfg.max_tracks=1; Engine engine(cfg); auto c=context(); const auto p=surface();
  const auto bytes=engine.working_bytes();
  for(int i=0;i<20000;++i) {
    c.image_ns+=100000000; c.cloud_ns=c.image_ns;
    auto d=detection(); d.label_id=uint32_t(i%5);
    if(i%17==0) { process(engine,c,{}, {d}); invalid(single(engine),Status::InsufficientSupport); }
    else { process(engine,c,p,{d}); require(single(engine).status==Status::Valid,"sustained observation valid"); }
    require(engine.working_bytes()==bytes,"preallocated storage must not grow across sustained replay");
    require(engine.measurements().size()<=cfg.max_detections && engine.projected().size()<=cfg.max_points,"bounded result storage");
  }
}
void configuration_rejection() {
  for(int field=0;field<6;++field) {
    auto cfg=configuration();
    if(field==0)cfg.max_points=0;
    if(field==1)cfg.max_width=0;
    if(field==2)cfg.min_cluster_fraction=0.5;
    if(field==3)cfg.calibration_sigma_m=NAN;
    if(field==4)cfg.occlusion_radius_px=-1;
    if(field==5)cfg.lidar_sigma_m=1e308;
    bool threw=false; try { Engine engine(cfg); } catch(const std::invalid_argument&) { threw=true; }
    require(threw,"unsafe configuration rejected before allocation/runtime");
  }
}
}  // namespace
int main() {
  const std::vector<std::pair<const char*,std::function<void()>>> tests={
    {"analytic_projection",projection_and_distortion},{"transform_contracts",transform_contracts},
    {"covariance_and_frame",covariance_and_output_frame},{"unique_support",support_deduplication},
    {"overlapping_objects",overlapping_objects},
    {"accelerator_failure",accelerator_failure},
    {"occlusion_and_ambiguity",occlusion_and_ambiguity},{"visibility_reference_parity",visibility_reference_parity},
    {"mask_contracts",masks_and_malformed_masks},
    {"calibration_and_tf",trust_and_missing_geometry},{"timestamp_limits",timestamp_limits},
    {"malformed_and_capacity",malformed_and_capacity},{"tracker_and_drops",tracker_and_drop_behavior},
    {"sustained_storage",sustained_capacity_stability},{"configuration_rejection",configuration_rejection}};
  int failures=0;
  for(const auto& test:tests) {
    try { test.second(); std::cout<<"PASS "<<test.first<<'\n'; }
    catch(const std::exception& e) { ++failures; std::cerr<<"FAIL "<<test.first<<": "<<e.what()<<'\n'; }
  }
  std::cout<<tests.size()-failures<<"/"<<tests.size()<<" tests passed\n";
  return failures?1:0;
}
