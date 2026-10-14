#include "go2_fusion_cpp/core.hpp"
#ifdef GO2_FUSION_HAS_CUDA
#include "go2_fusion_cpp/cuda_visibility.hpp"
#endif
#include "go2_fusion_cpp/msg/fusion_array.hpp"
#include "go2_semantic_msgs/msg/semantic_detection_array.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <stdexcept>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <unordered_map>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <yaml-cpp/yaml.h>

namespace {
using Clock = std::chrono::steady_clock;
using Status = go2_fusion::Status;
using Image = sensor_msgs::msg::Image;
using Cloud = sensor_msgs::msg::PointCloud2;
using Info = sensor_msgs::msg::CameraInfo;
using Header = std_msgs::msg::Header;
int64_t ns(const builtin_interfaces::msg::Time &t) {
  return int64_t(t.sec) * 1000000000LL + t.nanosec;
}
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
go2_fusion::Transform transform(const geometry_msgs::msg::Transform &t) {
  const auto &q = t.rotation;
  const double n = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (!std::isfinite(n) || std::abs(n - 1.0) > 1e-3)
    throw std::runtime_error("TF quaternion is not unit length");
  go2_fusion::Transform result;
  result.rotation = {
      1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w),
      2 * (q.x * q.z + q.y * q.w),     2 * (q.x * q.y + q.z * q.w),
      1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w),
      2 * (q.x * q.z - q.y * q.w),     2 * (q.y * q.z + q.x * q.w),
      1 - 2 * (q.x * q.x + q.y * q.y)};
  result.translation = {t.translation.x, t.translation.y, t.translation.z};
  if (!result.valid())
    throw std::runtime_error("TF contains invalid rigid geometry");
  return result;
}
struct ObjectInput {
  go2_fusion::Detection detection;
  std::string label;
  std::vector<uint8_t> mask;
};
struct Batch {
  Header header;
  std::vector<ObjectInput> objects;
  Status input_status = Status::Valid;
  std::string input_detail;
  double input_decode_ms = 0;
  Clock::time_point received = Clock::now();
};
struct CachedImage {
  Image::ConstSharedPtr image;
  bool payload_valid = false;
};
struct Calibration {
  bool ready = false;
  std::string error = "Measured calibration files are required";
  uint32_t width = 0, height = 0;
  std::array<double, 9> k{};
  std::array<double, 5> d{};
  std::string parent, camera;
  go2_fusion::Transform parent_from_camera;
};
} // namespace

class FusionNode final : public rclcpp::Node {
public:
  FusionNode() : Node("go2_fusion"), tf_(get_clock()), listener_(tf_) {
    const auto bounded_size = [this](const char *name, int64_t value,
                                     int64_t maximum) {
      const auto v = declare_parameter<int64_t>(name, value);
      if (v < 1 || v > maximum)
        throw std::invalid_argument(std::string(name) + " out of bounds");
      return size_t(v);
    };
    history_size_ = bounded_size("history_size", 16, 64);
    pending_size_ = bounded_size("pending_size", 2, 8);
    max_bytes_ = bounded_size("max_input_bytes", 8388608, 67108864);
    max_mask_bytes_ = bounded_size("max_mask_bytes", 16777216, 67108864);
    go2_fusion::Config cfg;
    cfg.max_points = bounded_size("max_points", 100000, 1000000);
    cfg.max_detections = bounded_size("max_detections", 64, 256);
    cfg.max_width = bounded_size("max_width", 1280, 4096);
    cfg.max_height = bounded_size("max_height", 720, 2160);
    max_points_ = cfg.max_points;
    max_detections_ = cfg.max_detections;
    max_width_ = cfg.max_width;
    max_height_ = cfg.max_height;
    const double skew = declare_parameter<double>("max_cloud_skew_ms", 50);
    timeout_ms_ = declare_parameter<double>("pending_timeout_ms", 50);
    if (!std::isfinite(skew) || skew < 0 || skew > 1000 ||
        !std::isfinite(timeout_ms_) || timeout_ms_ < 0 || timeout_ms_ > 1000)
      throw std::invalid_argument(
          "Skew and pending deadline must be finite in [0,1000] ms");
    cfg.max_skew_ns = int64_t(skew * 1e6);
    max_skew_ns_ = cfg.max_skew_ns;
    const auto backend = declare_parameter<std::string>("backend", "cpu");
    go2_fusion::Engine::VisibilityFunction accelerator;
    if (backend != "cpu") {
      if (backend != "cuda" && backend != "cuda_mapped")
        throw std::invalid_argument(
            "backend must be cpu, cuda, or cuda_mapped");
#ifdef GO2_FUSION_HAS_CUDA
      const auto mode =
          backend == "cuda_mapped"
              ? go2_fusion::CudaVisibility::TransferMode::MappedPinned
              : go2_fusion::CudaVisibility::TransferMode::ExplicitCopy;
      cuda_ = std::make_shared<go2_fusion::CudaVisibility>(cfg, mode);
      accelerator = [worker =
                         cuda_](std::vector<go2_fusion::Projected> &projected,
                                uint32_t width, uint32_t height) {
        if (!worker->run(projected, width, height) ||
            worker->size() != projected.size())
          return false;
        const auto *flags = worker->flags();
        if (!flags && !projected.empty())
          return false;
        for (size_t i = 0; i < projected.size(); ++i)
          projected[i].visible = flags[i] != 0;
        return true;
      };
#else
      throw std::invalid_argument("CUDA backend requested in CPU-only build");
#endif
    }
    engine_ = std::make_unique<go2_fusion::Engine>(cfg, std::move(accelerator));
    points_.reserve(max_points_);
    detections_.reserve(max_detections_);
    fixed_ = declare_parameter<std::string>("fixed_frame", "odom");
    camera_frame_ = declare_parameter<std::string>(
        "camera_frame", "front_camera_optical_frame");
    calibration_id_ = declare_parameter<std::string>("calibration_id", "");
    calibration_validated_ =
        declare_parameter<bool>("calibration_validated", false);
    timing_validated_ = declare_parameter<bool>("timing_validated", false);
    cloud_geometry_validated_ =
        declare_parameter<bool>("cloud_geometry_validated", false);
    latency_clock_validated_ =
        declare_parameter<bool>("acquisition_clock_validated", false);
    publish_overlay_ = declare_parameter<bool>("publish_overlay", false);
    load_calibration(declare_parameter<std::string>("intrinsics_file", ""),
                     declare_parameter<std::string>("extrinsics_file", ""));
    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(history_size_);
    image_sub_ = create_subscription<Image>(
        declare_parameter<std::string>("image_topic",
                                       "/camera/front/image_raw"),
        sensor_qos, [this](Image::ConstSharedPtr msg) {
          const size_t channels =
              msg->encoding == "rgb8" || msg->encoding == "bgr8"     ? 3
              : msg->encoding == "rgba8" || msg->encoding == "bgra8" ? 4
              : msg->encoding == "mono8"                             ? 1
                                                                     : 0;
          const bool valid =
              msg->data.size() <= max_bytes_ && channels && msg->width &&
              msg->height && msg->width <= max_width_ &&
              msg->height <= max_height_ &&
              msg->step >= uint64_t(msg->width) * channels &&
              uint64_t(msg->step) * msg->height == msg->data.size();
          if (publish_overlay_ && valid)
            images_.push_back({msg, true});
          else {
            auto metadata = std::make_shared<Image>();
            metadata->header = msg->header;
            metadata->width = msg->width;
            metadata->height = msg->height;
            metadata->encoding = msg->encoding;
            metadata->step = msg->step;
            metadata->is_bigendian = msg->is_bigendian;
            images_.push_back({metadata, valid});
          }
          while (images_.size() > history_size_)
            images_.pop_front();
          pump();
        });
    cloud_sub_ = create_subscription<Cloud>(
        declare_parameter<std::string>("cloud_topic", "/go2/lidar/points"),
        sensor_qos, [this](Cloud::ConstSharedPtr msg) {
          if (msg->data.size() > max_bytes_) {
            ++dropped_;
            return;
          }
          clouds_.push_back(msg);
          while (clouds_.size() > history_size_)
            clouds_.pop_front();
          pump();
        });
    info_sub_ = create_subscription<Info>(
        declare_parameter<std::string>("camera_info_topic",
                                       "/camera/front/camera_info"),
        sensor_qos, [this](Info::ConstSharedPtr msg) {
          if (msg->d.size() > 16) {
            ++dropped_;
            return;
          }
          infos_.push_back(msg);
          while (infos_.size() > history_size_)
            infos_.pop_front();
          pump();
        });
    output_ = create_publisher<go2_fusion_cpp::msg::FusionArray>(
        declare_parameter<std::string>("output_topic", "/fusion/objects"),
        rclcpp::QoS(2).reliable());
    overlay_ = create_publisher<Image>("/fusion/projected_overlay", sensor_qos);
    const auto input_type =
        declare_parameter<std::string>("input_type", "semantic");
    const auto standard_topic = declare_parameter<std::string>(
        "detections_topic", "/fusion/detections_2d");
    const auto semantic_topic = declare_parameter<std::string>(
        "semantic_detections_topic", "/semantic/detections");
    if (input_type == "vision")
      vision_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
          standard_topic, rclcpp::QoS(2).reliable(),
          [this](vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) {
            on_vision(*msg);
          });
    else if (input_type == "semantic")
      semantic_sub_ =
          create_subscription<go2_semantic_msgs::msg::SemanticDetectionArray>(
              semantic_topic, rclcpp::QoS(2).reliable(),
              [this](
                  go2_semantic_msgs::msg::SemanticDetectionArray::ConstSharedPtr
                      msg) { on_semantic(*msg); });
    else
      throw std::invalid_argument("input_type must be semantic or vision");
    timer_ =
        create_wall_timer(std::chrono::milliseconds(5), [this] { pump(); });
    parameter_guard_ = add_on_set_parameters_callback(
        [](const std::vector<rclcpp::Parameter> &parameters) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = true;
          for (const auto &parameter : parameters) {
            if (parameter.get_name() != "use_sim_time") {
              result.successful = false;
              result.reason = "Fusion configuration requires restart to "
                              "preserve calibration and buffer invariants";
              break;
            }
          }
          return result;
        });
    RCLCPP_INFO(get_logger(),
                "Isolated fusion ready; camera calibration=%s, cloud "
                "geometry=%s, timing=%s",
                calibration_.ready && calibration_validated_ ? "measured"
                                                             : "untrusted",
                cloud_geometry_validated_ ? "validated" : "untrusted",
                timing_validated_ ? "validated" : "untrusted");
  }

private:
  void load_calibration(const std::string &intrinsics,
                        const std::string &extrinsics) {
    try {
      if (intrinsics.empty() || extrinsics.empty())
        return;
      const auto i = YAML::LoadFile(intrinsics), e = YAML::LoadFile(extrinsics);
      if (!i["nominal"] || !e["nominal"] || i["nominal"].as<bool>() ||
          e["nominal"].as<bool>())
        throw std::runtime_error(
            "Both calibration files require explicit nominal:false");
      calibration_.width = i["image_width"].as<uint32_t>();
      calibration_.height = i["image_height"].as<uint32_t>();
      auto k = i["camera_matrix"]["data"].as<std::vector<double>>();
      auto d = i["distortion_coefficients"]["data"].as<std::vector<double>>();
      if (k.size() != 9 || d.size() != 5 ||
          i["distortion_model"].as<std::string>() != "plumb_bob")
        throw std::runtime_error(
            "Only five-coefficient plumb_bob calibration is supported");
      std::copy(k.begin(), k.end(), calibration_.k.begin());
      std::copy(d.begin(), d.end(), calibration_.d.begin());
      calibration_.parent = e["parent_frame"].as<std::string>();
      calibration_.camera = e["child_frame"].as<std::string>();
      auto xyz = e["xyz"].as<std::vector<double>>(),
           rpy = e["rpy"].as<std::vector<double>>();
      if (xyz.size() != 3 || rpy.size() != 3)
        throw std::runtime_error("Extrinsic xyz/rpy dimensions invalid");
      auto &t = calibration_.parent_from_camera;
      std::copy(xyz.begin(), xyz.end(), t.translation.begin());
      const double cr = cos(rpy[0]), sr = sin(rpy[0]), cp = cos(rpy[1]),
                   sp = sin(rpy[1]), cy = cos(rpy[2]), sy = sin(rpy[2]);
      t.rotation = {cy * cp,
                    cy * sp * sr - sy * cr,
                    cy * sp * cr + sy * sr,
                    sy * cp,
                    sy * sp * sr + cy * cr,
                    sy * sp * cr - cy * sr,
                    -sp,
                    cp * sr,
                    cp * cr};
      go2_fusion::Camera cam;
      cam.width = calibration_.width;
      cam.height = calibration_.height;
      cam.fx = k[0];
      cam.fy = k[4];
      cam.cx = k[2];
      cam.cy = k[5];
      cam.distortion = calibration_.d;
      cam.frame = calibration_.camera;
      if (!cam.valid() || !t.valid() || calibration_.parent.empty() ||
          calibration_.parent == calibration_.camera ||
          calibration_.camera != camera_frame_)
        throw std::runtime_error(
            "Calibration geometry or calibrated frame is invalid");
      for (double v : k)
        if (!std::isfinite(v))
          throw std::runtime_error("Nonfinite calibration K");
      if (k[1] != 0 || k[3] != 0 || k[6] != 0 || k[7] != 0 || k[8] != 1)
        throw std::runtime_error(
            "Camera K must have standard zero-skew pinhole form");
      calibration_.ready = true;
      calibration_.error.clear();
    } catch (const std::exception &e) {
      calibration_.error = e.what();
      RCLCPP_ERROR(get_logger(), "Calibration rejected: %s", e.what());
    }
  }
  uint32_t label_id(const std::string &label) {
    if (label.empty() || label.size() > 256)
      throw std::runtime_error("Invalid detector label");
    auto found = labels_.find(label);
    if (found != labels_.end())
      return found->second;
    if (labels_.size() >= 1024)
      throw std::length_error("Label dictionary capacity exceeded");
    const auto id = uint32_t(labels_.size() + 1);
    labels_.emplace(label, id);
    return id;
  }
  ObjectInput object(const vision_msgs::msg::BoundingBox2D &bbox,
                     const std::string &label, double score) {
    if (!std::isfinite(bbox.center.theta) || std::abs(bbox.center.theta) > 1e-8)
      throw std::runtime_error("Rotated detections are unsupported");
    ObjectInput o;
    o.label = label;
    auto &d = o.detection;
    d.x0 = bbox.center.position.x - bbox.size_x / 2;
    d.y0 = bbox.center.position.y - bbox.size_y / 2;
    d.x1 = bbox.center.position.x + bbox.size_x / 2;
    d.y1 = bbox.center.position.y + bbox.size_y / 2;
    d.score = score;
    d.label_id = label_id(label);
    if (!std::isfinite(d.x0) || !std::isfinite(d.y0) || !std::isfinite(d.x1) ||
        !std::isfinite(d.y1) || !std::isfinite(score) || score < 0 ||
        score > 1 || bbox.size_x <= 0 || bbox.size_y <= 0)
      throw std::runtime_error("Nonfinite or invalid detection geometry/score");
    return o;
  }
  void on_vision(const vision_msgs::msg::Detection2DArray &msg) {
    Batch b;
    b.header = msg.header;
    try {
      if (msg.detections.size() > max_detections_)
        throw std::length_error("Detection capacity exceeded");
      for (const auto &d : msg.detections) {
        if (ns(d.header.stamp) != ns(msg.header.stamp) ||
            d.header.frame_id != msg.header.frame_id || d.results.empty())
          throw std::runtime_error(
              "Detection headers/results do not match source image");
        const auto best =
            std::max_element(d.results.begin(), d.results.end(),
                             [](const auto &a, const auto &z) {
                               return a.hypothesis.score < z.hypothesis.score;
                             });
        b.objects.push_back(
            object(d.bbox, best->hypothesis.class_id, best->hypothesis.score));
      }
    } catch (const std::length_error &e) {
      b.input_status = Status::CapacityExceeded;
      b.input_detail = e.what();
    } catch (const std::exception &e) {
      b.input_status = Status::MalformedInput;
      b.input_detail = e.what();
    }
    b.input_decode_ms = elapsed_ms(b.received);
    enqueue(std::move(b));
  }
  void on_semantic(const go2_semantic_msgs::msg::SemanticDetectionArray &msg) {
    Batch b;
    b.header = msg.header;
    try {
      if (msg.detections.size() > max_detections_)
        throw std::length_error("Detection capacity exceeded");
      if (msg.source_frame_id != msg.header.frame_id)
        throw std::runtime_error("Semantic source frame mismatch");
      size_t mask_bytes = 0;
      for (const auto &d : msg.detections) {
        auto o = object(d.bbox_2d, d.label, d.score);
        if (!d.mask_rle.empty()) {
          const auto &roi = d.mask_roi;
          const uint64_t pixels = uint64_t(roi.width) * roi.height;
          if (!roi.width || !roi.height ||
              uint64_t(roi.x_offset) + roi.width > max_width_ ||
              uint64_t(roi.y_offset) + roi.height > max_height_)
            throw std::runtime_error(
                "Mask ROI is outside supported image bounds");
          if (pixels > max_mask_bytes_ ||
              mask_bytes + pixels > max_mask_bytes_ ||
              d.mask_rle.size() > max_mask_bytes_)
            throw std::length_error("Mask capacity exceeded");
          mask_bytes += pixels;
          o.mask.resize(pixels);
          size_t cursor = 0;
          uint8_t value = 0;
          for (uint8_t count : d.mask_rle) {
            if (cursor + count > pixels)
              throw std::runtime_error("Mask RLE overrun");
            std::fill(o.mask.begin() + cursor, o.mask.begin() + cursor + count,
                      value);
            cursor += count;
            value ^= 1;
          }
          if (cursor != pixels)
            throw std::runtime_error("Mask RLE underrun");
          o.detection.mask_x = roi.x_offset;
          o.detection.mask_y = roi.y_offset;
          o.detection.mask_width = roi.width;
          o.detection.mask_height = roi.height;
        }
        b.objects.push_back(std::move(o));
      }
    } catch (const std::length_error &e) {
      b.input_status = Status::CapacityExceeded;
      b.input_detail = e.what();
    } catch (const std::exception &e) {
      b.input_status = Status::MalformedInput;
      b.input_detail = e.what();
    }
    b.input_decode_ms = elapsed_ms(b.received);
    enqueue(std::move(b));
  }
  void enqueue(Batch b) {
    if (pending_.size() >= pending_size_) {
      ++dropped_;
      publish(pending_.front(), Status::CapacityExceeded,
              "Pending batch replaced", nullptr, nullptr, 0);
      pending_.pop_front();
    }
    pending_.push_back(std::move(b));
    pump();
  }
  void pump() {
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (process(*it))
        it = pending_.erase(it);
      else
        ++it;
    }
  }
  bool waiting(Batch &b, Status s, const std::string &detail) {
    if (elapsed_ms(b.received) < timeout_ms_)
      return false;
    publish(b, s, detail, nullptr, nullptr, 0);
    return true;
  }
  void parse_cloud(const Cloud &cloud) {
    points_.clear();
    const uint64_t count = uint64_t(cloud.width) * cloud.height;
    if (count > max_points_)
      throw std::length_error("Point capacity exceeded");
    if (!count || cloud.is_bigendian || !cloud.point_step ||
        uint64_t(cloud.width) * cloud.point_step > cloud.row_step ||
        uint64_t(cloud.row_step) * cloud.height != cloud.data.size())
      throw std::runtime_error(
          "Invalid PointCloud2 dimensions, layout, or byte order");
    std::array<sensor_msgs::msg::PointField, 3> fields;
    const std::array<std::string, 3> names = {"x", "y", "z"};
    for (size_t axis = 0; axis < 3; ++axis) {
      size_t matches = 0;
      for (const auto &f : cloud.fields)
        if (f.name == names[axis]) {
          fields[axis] = f;
          ++matches;
        }
      const auto &f = fields[axis];
      const size_t size =
          f.datatype == sensor_msgs::msg::PointField::FLOAT32   ? 4
          : f.datatype == sensor_msgs::msg::PointField::FLOAT64 ? 8
                                                                : 0;
      if (matches != 1 || f.count != 1 || !size ||
          uint64_t(f.offset) + size > cloud.point_step)
        throw std::runtime_error(
            "Cloud needs unique scalar float x/y/z fields within point_step");
    }
    for (uint32_t row = 0; row < cloud.height; ++row)
      for (uint32_t col = 0; col < cloud.width; ++col) {
        float xyz[3];
        const auto *base = cloud.data.data() + size_t(row) * cloud.row_step +
                           size_t(col) * cloud.point_step;
        for (size_t axis = 0; axis < 3; ++axis) {
          if (fields[axis].datatype == sensor_msgs::msg::PointField::FLOAT32)
            std::memcpy(&xyz[axis], base + fields[axis].offset, 4);
          else {
            double v;
            std::memcpy(&v, base + fields[axis].offset, 8);
            xyz[axis] = float(v);
          }
        }
        points_.push_back({xyz[0], xyz[1], xyz[2]});
      }
  }
  bool process(Batch &b) {
    const auto start = Clock::now();
    if (b.input_status != Status::Valid) {
      publish(b, b.input_status, b.input_detail, nullptr, nullptr, 0);
      return true;
    }
    if (ns(b.header.stamp) <= 0 || b.header.frame_id != camera_frame_) {
      publish(b, Status::MalformedInput, "Detection source stamp/frame invalid",
              nullptr, nullptr, 0);
      return true;
    }
    if (!calibration_validated_ || !timing_validated_ ||
        !cloud_geometry_validated_ || calibration_id_.empty() ||
        !calibration_.ready) {
      publish(b, Status::UntrustedCalibration,
              calibration_.error.empty()
                  ? "Calibration, upstream LiDAR geometry, and acquisition "
                    "timing require validated provenance"
                  : calibration_.error,
              nullptr, nullptr, 0);
      return true;
    }
    Image::ConstSharedPtr image;
    Info::ConstSharedPtr info;
    Cloud::ConstSharedPtr cloud;
    bool image_payload_valid = false;
    for (const auto &candidate : images_)
      if (ns(candidate.image->header.stamp) == ns(b.header.stamp) &&
          candidate.image->header.frame_id == b.header.frame_id) {
        image = candidate.image;
        image_payload_valid = candidate.payload_valid;
      }
    for (const auto &candidate : infos_)
      if (ns(candidate->header.stamp) == ns(b.header.stamp) &&
          candidate->header.frame_id == b.header.frame_id)
        info = candidate;
    int64_t gap = std::numeric_limits<int64_t>::max();
    for (const auto &candidate : clouds_) {
      const auto delta =
          std::abs(ns(candidate->header.stamp) - ns(b.header.stamp));
      if (delta < gap) {
        gap = delta;
        cloud = candidate;
      }
    }
    if (!image || !info)
      return waiting(b, Status::TimestampSkew,
                     "Exact stamped source image and CameraInfo unavailable");
    if (!cloud || gap > max_skew_ns_)
      return waiting(b, Status::TimestampSkew,
                     "No cloud within bounded timestamp skew");
    if (!image_payload_valid || image->width != calibration_.width ||
        image->height != calibration_.height || info->width != image->width ||
        info->height != image->height ||
        info->distortion_model != "plumb_bob" || info->d.size() != 5 ||
        info->binning_x > 1 || info->binning_y > 1 || info->roi.width ||
        info->roi.height) {
      publish(b, Status::MalformedInput,
              "Image/CameraInfo dimensions, model, ROI or payload mismatch",
              cloud.get(), nullptr, elapsed_ms(start));
      return true;
    }
    for (size_t i = 0; i < 9; ++i)
      if (!std::isfinite(info->k[i]) ||
          std::abs(info->k[i] - calibration_.k[i]) > 1e-7) {
        publish(b, Status::UntrustedCalibration,
                "CameraInfo K differs from measured calibration", cloud.get(),
                nullptr, elapsed_ms(start));
        return true;
      }
    for (size_t i = 0; i < 5; ++i)
      if (!std::isfinite(info->d[i]) ||
          std::abs(info->d[i] - calibration_.d[i]) > 1e-7) {
        publish(b, Status::UntrustedCalibration,
                "CameraInfo D differs from measured calibration", cloud.get(),
                nullptr, elapsed_ms(start));
        return true;
      }
    go2_fusion::Context context;
    context.image_ns = ns(b.header.stamp);
    context.cloud_ns = ns(cloud->header.stamp);
    context.output_frame = fixed_;
    context.intrinsics_trusted = context.extrinsics_trusted =
        context.timing_trusted = true;
    try {
      const rclcpp::Time image_time(b.header.stamp,
                                    get_clock()->get_clock_type()),
          cloud_time(cloud->header.stamp, get_clock()->get_clock_type());
      if (cloud->header.frame_id.empty() || context.cloud_ns <= 0)
        throw std::runtime_error("Cloud stamp/frame invalid");
      context.camera_from_cloud = transform(
          tf_.lookupTransform(camera_frame_, image_time, cloud->header.frame_id,
                              cloud_time, fixed_, rclcpp::Duration(0, 0))
              .transform);
      context.output_from_camera =
          transform(tf_.lookupTransform(fixed_, camera_frame_, image_time,
                                        rclcpp::Duration(0, 0))
                        .transform);
      const auto measured =
          transform(tf_.lookupTransform(calibration_.parent, camera_frame_,
                                        image_time, rclcpp::Duration(0, 0))
                        .transform);
      for (size_t i = 0; i < 9; ++i)
        if (std::abs(measured.rotation[i] -
                     calibration_.parent_from_camera.rotation[i]) > 1e-5)
          throw std::domain_error(
              "TF rotation differs from measured extrinsics");
      for (size_t i = 0; i < 3; ++i)
        if (std::abs(measured.translation[i] -
                     calibration_.parent_from_camera.translation[i]) > 1e-5)
          throw std::domain_error(
              "TF translation differs from measured extrinsics");
      context.tf_valid = true;
    } catch (const tf2::TransformException &e) {
      return waiting(b, Status::MissingTransform, e.what());
    } catch (const std::domain_error &e) {
      publish(b, Status::UntrustedCalibration, e.what(), cloud.get(), nullptr,
              elapsed_ms(start));
      return true;
    } catch (const std::exception &e) {
      publish(b, Status::MalformedInput, e.what(), cloud.get(), nullptr,
              elapsed_ms(start));
      return true;
    }
    try {
      parse_cloud(*cloud);
    } catch (const std::length_error &e) {
      publish(b, Status::CapacityExceeded, e.what(), cloud.get(), nullptr,
              elapsed_ms(start));
      return true;
    } catch (const std::exception &e) {
      publish(b, Status::MalformedInput, e.what(), cloud.get(), nullptr,
              elapsed_ms(start));
      return true;
    }
    try {
      detections_.clear();
      for (auto &o : b.objects) {
        o.detection.mask = o.mask.empty() ? nullptr : o.mask.data();
        o.detection.mask_size = o.mask.size();
        detections_.push_back(o.detection);
      }
      go2_fusion::Camera camera;
      camera.width = image->width;
      camera.height = image->height;
      camera.fx = info->k[0];
      camera.fy = info->k[4];
      camera.cx = info->k[2];
      camera.cy = info->k[5];
      camera.frame = camera_frame_;
      camera.distortion = calibration_.d;
      const auto status =
          engine_->process(camera, context, points_.data(), points_.size(),
                           detections_.data(), detections_.size());
      std::string status_detail = go2_fusion::status_name(status);
#ifdef GO2_FUSION_HAS_CUDA
      if (status == Status::BackendError && cuda_)
        status_detail += " " + cuda_->error();
#endif
      publish(b, status, status_detail, cloud.get(), &engine_->measurements(),
              elapsed_ms(start));
      if (publish_overlay_ && status == Status::Valid)
        overlay(*image);
    } catch (const std::length_error &e) {
      publish(b, Status::CapacityExceeded, e.what(), cloud.get(), nullptr,
              elapsed_ms(start));
    } catch (const std::exception &e) {
      engine_->reset_tracks();
      publish(b, Status::BackendError, e.what(), cloud.get(), nullptr,
              elapsed_ms(start));
    }
    return true;
  }
  void publish(const Batch &b, Status status, const std::string &detail,
               const Cloud *cloud,
               const std::vector<go2_fusion::Measurement> *measurements,
               double adapter_ms) {
    go2_fusion_cpp::msg::FusionArray out;
    out.header = b.header;
    out.header.frame_id = fixed_;
    out.camera_frame = b.header.frame_id;
    out.calibration_id = calibration_id_;
    out.status = uint8_t(status);
    out.status_detail = detail;
    out.fusion_compute_ms = measurements ? engine_->timing().total_ms : -1;
    out.adapter_compute_ms = adapter_ms + b.input_decode_ms;
    out.synchronization_wait_ms =
        std::max(0.0, elapsed_ms(b.received) - out.adapter_compute_ms);
    out.dropped_pending_batches = dropped_;
    out.sensor_to_output_ms = -1;
    out.sensor_latency_valid = false;
    if (cloud) {
      out.cloud_stamp = cloud->header.stamp;
      out.cloud_frame = cloud->header.frame_id;
      out.image_cloud_skew_ms =
          double(ns(cloud->header.stamp) - ns(b.header.stamp)) / 1e6;
    } else
      out.image_cloud_skew_ms = std::numeric_limits<double>::quiet_NaN();
    const auto now = get_clock()->now().nanoseconds();
    const int64_t oldest_stamp =
        cloud ? std::min(ns(b.header.stamp), ns(cloud->header.stamp))
              : ns(b.header.stamp);
    if (latency_clock_validated_ && timing_validated_ &&
        !get_parameter("use_sim_time").as_bool() && oldest_stamp > 0 &&
        now >= ns(b.header.stamp) &&
        (!cloud || now >= ns(cloud->header.stamp))) {
      out.sensor_to_output_ms = double(now - oldest_stamp) / 1e6;
      out.sensor_latency_valid = true;
    }
    out.objects.reserve(b.objects.size());
    for (size_t i = 0; i < b.objects.size(); ++i) {
      go2_fusion_cpp::msg::FusionObject o;
      o.label = b.objects[i].label;
      o.detector_score = b.objects[i].detection.score;
      o.position.x = o.position.y = o.position.z =
          std::numeric_limits<double>::quiet_NaN();
      o.covariance.fill(std::numeric_limits<double>::quiet_NaN());
      o.status = uint8_t(status);
      o.status_detail = detail;
      o.valid = false;
      if (measurements && i < measurements->size()) {
        const auto &m = (*measurements)[i];
        o.status = uint8_t(m.status);
        o.status_detail = go2_fusion::status_name(m.status);
        o.support_points = m.support;
        o.valid = m.status == Status::Valid;
        if (o.valid) {
          o.track_id = m.track_id;
          o.position.x = m.position.x;
          o.position.y = m.position.y;
          o.position.z = m.position.z;
          o.covariance = m.covariance;
        }
      }
      out.objects.push_back(std::move(o));
    }
    output_->publish(out);
  }
  void overlay(const Image &source) {
    if ((source.encoding != "rgb8" && source.encoding != "bgr8") ||
        source.step < size_t(source.width) * 3)
      return;
    Image out = source;
    const bool rgb = source.encoding == "rgb8";
    for (const auto &p : engine_->projected())
      if (p.visible) {
        const int u = int(p.u), v = int(p.v);
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (u + dx < 0 || v + dy < 0 || u + dx >= int(source.width) ||
                v + dy >= int(source.height))
              continue;
            const auto offset = size_t(v + dy) * out.step + size_t(u + dx) * 3;
            out.data[offset + (rgb ? 0 : 2)] =
                uint8_t(std::min(255.0f, p.camera_point.z * 32));
            out.data[offset + 1] = 255;
            out.data[offset + (rgb ? 2 : 0)] = 0;
          }
      }
    overlay_->publish(out);
  }
  size_t history_size_, pending_size_, max_bytes_, max_mask_bytes_, max_points_,
      max_detections_, max_width_, max_height_;
  int64_t max_skew_ns_;
  double timeout_ms_;
  uint64_t dropped_ = 0;
  bool calibration_validated_, timing_validated_, cloud_geometry_validated_,
      latency_clock_validated_, publish_overlay_;
  std::string fixed_, camera_frame_, calibration_id_;
  Calibration calibration_;
  std::unordered_map<std::string, uint32_t> labels_;
#ifdef GO2_FUSION_HAS_CUDA
  std::shared_ptr<go2_fusion::CudaVisibility> cuda_;
#endif
  std::unique_ptr<go2_fusion::Engine> engine_;
  std::vector<go2_fusion::Point> points_;
  std::vector<go2_fusion::Detection> detections_;
  std::deque<CachedImage> images_;
  std::deque<Cloud::ConstSharedPtr> clouds_;
  std::deque<Info::ConstSharedPtr> infos_;
  std::deque<Batch> pending_;
  tf2_ros::Buffer tf_;
  tf2_ros::TransformListener listener_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_;
  rclcpp::Subscription<Cloud>::SharedPtr cloud_sub_;
  rclcpp::Subscription<Info>::SharedPtr info_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr
      vision_sub_;
  rclcpp::Subscription<
      go2_semantic_msgs::msg::SemanticDetectionArray>::SharedPtr semantic_sub_;
  rclcpp::Publisher<go2_fusion_cpp::msg::FusionArray>::SharedPtr output_;
  rclcpp::Publisher<Image>::SharedPtr overlay_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameter_guard_;
};
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<FusionNode>());
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("go2_fusion"), "Fusion stopped: %s",
                 e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
