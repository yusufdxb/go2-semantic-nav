#pragma once
#include <array>
#include <cstddef>
#include <cmath>
#include <functional>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace go2_fusion {
struct Point { float x, y, z; };
struct Transform {
  std::array<double,9> rotation{1,0,0,0,1,0,0,0,1};
  std::array<double,3> translation{0,0,0};
  bool valid() const;
  Point apply(Point p) const;
};
struct Camera {
  uint32_t width=1280, height=720;
  double fx=537, fy=537, cx=639.5, cy=359.5;
  std::array<double,5> distortion{};
  std::string frame="front_camera_optical_frame";
  bool valid() const;
};
struct Detection {
  float x0=0,y0=0,x1=0,y1=0,score=0;
  uint32_t label_id=0;
  // Optional full-image row-major binary mask; lifetime covers process().
  const uint8_t* mask=nullptr;
  size_t mask_size=0;
  uint32_t mask_x=0,mask_y=0,mask_width=0,mask_height=0;
};
enum class Status : uint8_t {
  Valid, UntrustedCalibration, MissingTransform, TimestampSkew,
  MalformedInput, CapacityExceeded, InsufficientSupport, AmbiguousDepth,
  OutOfOrder, BackendError
};
const char* status_name(Status s);
struct Context {
  int64_t image_ns=0,cloud_ns=0;
  bool intrinsics_trusted=false,extrinsics_trusted=false,timing_trusted=false;
  bool tf_valid=false;
  Transform camera_from_cloud, output_from_camera;
  std::string output_frame="odom";
};
struct Measurement {
  Status status=Status::InsufficientSupport;
  uint32_t label_id=0,support=0;
  uint64_t track_id=0;
  int64_t stamp_ns=0;
  Point position{NAN,NAN,NAN};
  std::array<double,9> covariance{NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN};
};
struct Timing { double projection_ms=0,visibility_ms=0,association_ms=0,total_ms=0; };
struct Config {
  size_t max_points=100000,max_detections=64,max_tracks=128;
  uint32_t max_width=1280,max_height=720;
  int occlusion_radius_px=10;
  double occlusion_margin_m=0.3,min_depth_m=0.2,max_depth_m=8,max_normalized_radius=1.3;
  uint32_t min_support=6;
  double cluster_gap_m=0.25,min_cluster_fraction=0.55;
  double lidar_sigma_m=0.03,calibration_sigma_m=0.05,max_object_speed_mps=2;
  int64_t max_skew_ns=50000000,track_ttl_ns=500000000;
  double track_gate_m=0.6;
};
struct Projected { Point camera_point; float u=0,v=0; uint32_t pixel=0; bool visible=false; int32_t owner=-1; };
class Engine {
 public:
  using VisibilityFunction=std::function<bool(std::vector<Projected>&,uint32_t,uint32_t)>;
  explicit Engine(Config cfg={},VisibilityFunction accelerator={});
  // Single-owner engine. Storage returned by accessors lasts until next call.
  Status process(const Camera&,const Context&,const Point*,size_t,const Detection*,size_t);
  const std::vector<Measurement>& measurements() const { return results_; }
  const std::vector<Projected>& projected() const { return projected_; }
  const Timing& timing() const { return timing_; }
  size_t working_bytes() const;
  void reset_tracks();
 private:
  struct Track { uint64_t id; uint32_t label; Point position; int64_t stamp; bool used; };
  Config cfg_;
  VisibilityFunction accelerator_;
  std::vector<Projected> projected_;
  std::vector<float> depth_,scratch_,nearest_;
  std::vector<int> deque_;
  std::vector<uint32_t> candidates_;
  std::vector<Measurement> results_;
  std::vector<Track> tracks_;
  Timing timing_;
  uint64_t next_id_=1;
  int64_t last_stamp_=0;
  std::string last_frame_;
  void visibility(uint32_t width,uint32_t height);
  Measurement associate(const Detection&,size_t,const Camera&,const Context&);
  void track(const Context&);
};
}  // namespace go2_fusion
