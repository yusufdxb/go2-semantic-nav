#include <gst/gst.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "go2_front_camera_cpp/pipeline.hpp"

using go2_front_camera::build_pipeline;
using go2_front_camera::PipelineConfig;

static bool has(const std::string & s, const std::string & sub) { return s.find(sub) != std::string::npos; }

TEST(Pipeline, DefaultIsGo2MulticastWithLowLatencySoftwareDecode)
{
  const auto p = build_pipeline(PipelineConfig{});
  EXPECT_TRUE(has(p, "udpsrc address=230.1.1.1 port=1720 buffer-size=8388608 caps="));
  EXPECT_FALSE(has(p, "multicast-iface"));
  EXPECT_FALSE(has(p, "rtpjitterbuffer"));
  EXPECT_TRUE(has(p, "avdec_h264 thread-type=slice output-corrupt=false"));
  EXPECT_TRUE(has(p, "appsink name=sink drop=true max-buffers=1 sync=false"));
}

TEST(Pipeline, JetsonDecoderDisablesDpb)
{
  PipelineConfig c;
  c.decoder = "nvv4l2";
  c.multicast_iface = "enP8p1s0";
  const auto p = build_pipeline(c);
  EXPECT_TRUE(has(p, "multicast-iface=enP8p1s0"));
  EXPECT_TRUE(has(p, "nvv4l2decoder disable-dpb=true enable-max-performance=true ! nvvidconv"));
}

TEST(Pipeline, ScaleAndRejections)
{
  PipelineConfig c;
  c.output_width = 640;
  c.output_height = 360;
  EXPECT_TRUE(has(build_pipeline(c), "videoscale ! video/x-raw,format=BGR,width=640,height=360 ! appsink"));
  c.output_height = 0;
  EXPECT_THROW(build_pipeline(c), std::invalid_argument);
  PipelineConfig d;
  d.decoder = "ffmpeg";
  EXPECT_THROW(build_pipeline(d), std::invalid_argument);
  PipelineConfig e;
  e.port = 0;
  EXPECT_THROW(build_pipeline(e), std::invalid_argument);
}

// Every software-decode variant must parse with the installed GStreamer (the
// Jetson decoder variant needs Jetson plugins and is checked on the robot).
TEST(Pipeline, SoftwareVariantsParse)
{
  gst_init(nullptr, nullptr);
  for (int scaled = 0; scaled < 2; ++scaled) {
    PipelineConfig c;
    c.address = "127.0.0.1";
    c.port = 56299;
    if (scaled) {
      c.output_width = 960;
      c.output_height = 540;
    }
    GError * err = nullptr;
    GstElement * p = gst_parse_launch(build_pipeline(c).c_str(), &err);
    EXPECT_EQ(err, nullptr) << (err ? err->message : "") << " in: " << build_pipeline(c);
    if (err) {
      g_error_free(err);
    }
    ASSERT_NE(p, nullptr);
    GstElement * sink = gst_bin_get_by_name(GST_BIN(p), "sink");
    EXPECT_NE(sink, nullptr);
    if (sink) {
      gst_object_unref(sink);
    }
    gst_object_unref(p);
  }
}
