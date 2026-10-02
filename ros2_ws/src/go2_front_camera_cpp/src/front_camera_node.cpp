// front_camera_node (C++): GO2 front camera RTP H.264 multicast -> sensor_msgs/Image.
//
// Stamps: udpsrc timestamps each packet with the pipeline running time at
// arrival, and the depayloader carries it to the decoded frame. The image is
// stamped with the ROS time of that ARRIVAL (minus `latency_s`, the
// capture-to-arrival delay measured in the lab), not the publish time, so
// decode and conversion time do not become timestamp error against the LiDAR.
//
// /camera/front/latency (std_msgs/String, JSON, 1 Hz): fps and
// arrival->publish p50/p95/max in ms over the last second, restarts, frames.
//
// If no frame arrives for `restart_after_s` the pipeline is rebuilt: a
// decoder that stops after the stream restarts looks the same as a camera
// that went away. Receive-only: joining a multicast group sends nothing to
// the robot.

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "go2_front_camera_cpp/pipeline.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

namespace go2_front_camera {

class FrontCameraNode : public rclcpp::Node {
public:
  FrontCameraNode() : rclcpp::Node("go2_front_camera")
  {
    PipelineConfig cfg;
    cfg.address = declare_parameter<std::string>("address", cfg.address);
    cfg.port = static_cast<int>(declare_parameter<int64_t>("port", cfg.port));
    cfg.multicast_iface = declare_parameter<std::string>("multicast_iface", "");
    cfg.buffer_size = static_cast<int>(declare_parameter<int64_t>("buffer_size", cfg.buffer_size));
    cfg.decoder = declare_parameter<std::string>("decoder", cfg.decoder);
    cfg.output_width = static_cast<int>(declare_parameter<int64_t>("output_width", 0));
    cfg.output_height = static_cast<int>(declare_parameter<int64_t>("output_height", 0));
    const auto override_desc = declare_parameter<std::string>("pipeline", "");
    frame_id_ = declare_parameter<std::string>("frame_id", "front_camera_optical_frame");
    latency_ns_ = static_cast<int64_t>(declare_parameter<double>("latency_s", 0.0) * 1e9);
    const double max_rate = declare_parameter<double>("max_rate_hz", 0.0);
    min_period_ns_ = max_rate > 0 ? static_cast<int64_t>(1e9 / max_rate) : 0;
    restart_after_ = std::chrono::duration<double>(declare_parameter<double>("restart_after_s", 3.0));
    const auto image_topic = declare_parameter<std::string>("image_topic", "/camera/front/image_raw");
    const auto stats_topic = declare_parameter<std::string>("stats_topic", "/camera/front/latency");

    desc_ = override_desc.empty() ? build_pipeline(cfg) : override_desc;
    // Depth 1, best effort: a late frame is worth less than the next one.
    image_pub_ = create_publisher<sensor_msgs::msg::Image>(image_topic, rclcpp::SensorDataQoS().keep_last(1));
    stats_pub_ = create_publisher<std_msgs::msg::String>(stats_topic, 10);
    stats_timer_ = create_wall_timer(std::chrono::seconds(1), [this] { publish_stats(); });

    RCLCPP_INFO(get_logger(), "front camera pipeline: %s", desc_.c_str());
    worker_ = std::thread([this] { run(); });
  }

  ~FrontCameraNode() override
  {
    stop_ = true;
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  void run()
  {
    while (!stop_) {
      GError * err = nullptr;
      GstElement * pipeline = gst_parse_launch(desc_.c_str(), &err);
      if (pipeline == nullptr || err != nullptr) {
        RCLCPP_FATAL(get_logger(), "bad pipeline: %s", err ? err->message : "unknown");
        if (err) {
          g_error_free(err);
        }
        rclcpp::shutdown();
        return;
      }
      GstElement * sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
      gst_element_set_state(pipeline, GST_STATE_PLAYING);
      auto last_frame = std::chrono::steady_clock::now();
      while (!stop_) {
        GstSample * sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 200 * GST_MSECOND);
        const auto now = std::chrono::steady_clock::now();
        if (sample == nullptr) {
          if (now - last_frame > restart_after_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                                 "no camera frame for %.1f s; rebuilding pipeline", restart_after_.count());
            ++restarts_;
            break;
          }
          continue;
        }
        last_frame = now;
        handle(pipeline, sample);
        gst_sample_unref(sample);
      }
      gst_element_set_state(pipeline, GST_STATE_NULL);
      gst_object_unref(sink);
      gst_object_unref(pipeline);
    }
  }

  void handle(GstElement * pipeline, GstSample * sample)
  {
    // Age of the frame since its packets arrived, on the pipeline clock.
    GstBuffer * buf = gst_sample_get_buffer(sample);
    int64_t age_ns = 0;
    GstClock * clock = gst_element_get_clock(pipeline);
    if (clock != nullptr && GST_BUFFER_PTS_IS_VALID(buf)) {
      const GstClockTime running = gst_clock_get_time(clock) - gst_element_get_base_time(pipeline);
      if (running > GST_BUFFER_PTS(buf)) {
        age_ns = static_cast<int64_t>(running - GST_BUFFER_PTS(buf));
      }
    }
    if (clock != nullptr) {
      gst_object_unref(clock);
    }
    const rclcpp::Time now = this->now();
    if (min_period_ns_ > 0 && (now - last_pub_).nanoseconds() < min_period_ns_) {
      return;
    }

    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, gst_sample_get_caps(sample))) {
      return;
    }
    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
      return;
    }
    const int w = GST_VIDEO_INFO_WIDTH(&info);
    const int h = GST_VIDEO_INFO_HEIGHT(&info);
    const size_t stride = static_cast<size_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 0));
    const size_t row = static_cast<size_t>(w) * 3;

    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header.stamp = now - rclcpp::Duration::from_nanoseconds(age_ns + latency_ns_);
    msg->header.frame_id = frame_id_;
    msg->width = static_cast<uint32_t>(w);
    msg->height = static_cast<uint32_t>(h);
    msg->encoding = "bgr8";
    msg->is_bigendian = 0;
    msg->step = static_cast<uint32_t>(row);
    if (stride == row) {
      msg->data.assign(map.data, map.data + row * h);
    } else {
      msg->data.resize(row * h);
      for (int y = 0; y < h; ++y) {
        std::memcpy(msg->data.data() + row * y, map.data + stride * y, row);
      }
    }
    gst_buffer_unmap(buf, &map);
    image_pub_->publish(std::move(msg));
    last_pub_ = now;

    const double total_ms = (age_ns + (this->now() - now).nanoseconds()) / 1e6;
    std::lock_guard<std::mutex> lock(stats_mutex_);
    window_ms_.push_back(total_ms);
    ++frames_;
  }

  void publish_stats()
  {
    std::vector<double> w;
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      w.swap(window_ms_);
    }
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(2);
    s << "{\"fps\": " << w.size() << ", \"frames\": " << frames_.load() << ", \"restarts\": " << restarts_.load();
    if (!w.empty()) {
      std::sort(w.begin(), w.end());
      auto q = [&w](double f) { return w[std::min(w.size() - 1, static_cast<size_t>(f * (w.size() - 1) + 0.5))]; };
      s << ", \"arrival_to_publish_ms\": {\"p50\": " << q(0.5) << ", \"p95\": " << q(0.95) << ", \"max\": " << w.back()
        << "}";
    }
    s << "}";
    std_msgs::msg::String m;
    m.data = s.str();
    stats_pub_->publish(m);
  }

  std::string desc_;
  std::string frame_id_;
  int64_t latency_ns_{0};
  int64_t min_period_ns_{0};
  std::chrono::duration<double> restart_after_{3.0};
  rclcpp::Time last_pub_{0, 0, RCL_ROS_TIME};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr stats_pub_;
  rclcpp::TimerBase::SharedPtr stats_timer_;
  std::thread worker_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> frames_{0};
  std::atomic<uint64_t> restarts_{0};
  std::mutex stats_mutex_;
  std::vector<double> window_ms_;
};

}  // namespace go2_front_camera

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // After rclcpp::init: GStreamer is only initialised once ROS is up.
  gst_init(nullptr, nullptr);
  auto node = std::make_shared<go2_front_camera::FrontCameraNode>();
  rclcpp::spin(node);
  node.reset();
  rclcpp::shutdown();
  return 0;
}
