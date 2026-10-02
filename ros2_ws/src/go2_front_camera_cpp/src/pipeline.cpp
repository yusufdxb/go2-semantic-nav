#include "go2_front_camera_cpp/pipeline.hpp"

#include <sstream>
#include <stdexcept>

namespace go2_front_camera {

std::string build_pipeline(const PipelineConfig & cfg)
{
  if (cfg.port <= 0 || cfg.port > 65535) {
    throw std::invalid_argument("port out of range");
  }
  if ((cfg.output_width == 0) != (cfg.output_height == 0) || cfg.output_width < 0 || cfg.output_height < 0) {
    throw std::invalid_argument("output_width and output_height must both be 0 or both be positive");
  }
  std::ostringstream p;
  p << "udpsrc address=" << cfg.address << " port=" << cfg.port;
  if (cfg.buffer_size > 0) {
    p << " buffer-size=" << cfg.buffer_size;
  }
  if (!cfg.multicast_iface.empty()) {
    p << " multicast-iface=" << cfg.multicast_iface;
  }
  // No rtpjitterbuffer: its default 200 ms latency is the opposite of the
  // goal, and reordering does not happen on the robot's own switch.
  p << " caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264\""
    << " ! rtph264depay ! h264parse";
  if (cfg.decoder == "avdec") {
    p << " ! avdec_h264 thread-type=slice output-corrupt=false ! videoconvert";
  } else if (cfg.decoder == "nvv4l2") {
    p << " ! nvv4l2decoder disable-dpb=true enable-max-performance=true"
      << " ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert";
  } else {
    throw std::invalid_argument("decoder must be 'avdec' or 'nvv4l2', got '" + cfg.decoder + "'");
  }
  // One caps filter: gst_parse_launch rejects two adjacent caps strings.
  if (cfg.output_width > 0) {
    p << " ! videoscale ! video/x-raw,format=BGR,width=" << cfg.output_width << ",height=" << cfg.output_height;
  } else {
    p << " ! video/x-raw,format=BGR";
  }
  p << " ! appsink name=sink drop=true max-buffers=1 sync=false";
  return p.str();
}

}  // namespace go2_front_camera
