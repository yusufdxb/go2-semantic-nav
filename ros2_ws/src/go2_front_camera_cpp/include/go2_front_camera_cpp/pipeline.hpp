// GStreamer pipeline description for the GO2 front camera RTP H.264 stream.
#pragma once

#include <string>

namespace go2_front_camera {

struct PipelineConfig {
  std::string address = "230.1.1.1";
  int port = 1720;
  std::string multicast_iface;  // empty: OS default route
  // Socket receive buffer. A 720p keyframe is hundreds of datagrams; with the
  // kernel default (~208 KB) a burst overflows it, one lost packet drops the
  // whole frame, and frames vanish silently. The kernel caps this at
  // net.core.rmem_max, which must be raised to match (the lab script does).
  int buffer_size = 8 * 1024 * 1024;
  // "avdec": libav software decode, slice threading (frame threading holds
  //          frames), corrupt frames dropped.
  // "nvv4l2": Jetson hardware decode with the decoded-picture buffer disabled
  //          (otherwise each frame comes out one decode late) and the
  //          decoder clocks pinned high. Valid only for streams without
  //          B-frame reordering.
  std::string decoder = "avdec";
  int output_width = 0;  // 0: native size
  int output_height = 0;
};

// Returns the pipeline, ending in `appsink name=sink` producing BGR.
// Throws std::invalid_argument on an unknown decoder or a bad size.
std::string build_pipeline(const PipelineConfig & cfg);

}  // namespace go2_front_camera
