"""front_camera_node against a local RTP H.264 sender (unicast loopback, no robot, no LAN traffic)."""

import shutil
import subprocess
import time

import pytest
import rclpy
from go2_rgb_lidar.front_camera_node import FrontCameraNode, build_pipeline
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

PORT = 56123
_ELEMENTS = ("openh264enc", "rtph264pay", "rtph264depay", "h264parse", "avdec_h264", "udpsrc", "udpsink")


def _have_elements() -> bool:
    if shutil.which("gst-inspect-1.0") is None or shutil.which("gst-launch-1.0") is None:
        return False
    return all(subprocess.run(["gst-inspect-1.0", "--exists", e]).returncode == 0 for e in _ELEMENTS)


pytestmark = pytest.mark.skipif(not _have_elements(), reason="GStreamer H.264 elements not installed")


def _sender(width, height):
    # A separate process, like the robot: the test process loads GStreamer only through the node.
    return subprocess.Popen(
        ["gst-launch-1.0", "-q", "videotestsrc", "is-live=true", "!",
         f"video/x-raw,width={width},height={height},framerate=15/1", "!", "videoconvert", "!",
         "openh264enc", "!", "rtph264pay", "config-interval=1", "pt=96", "!",
         "udpsink", "host=127.0.0.1", f"port={PORT}", "sync=false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _stop(proc):
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_build_pipeline_matches_go2_stream():
    desc = build_pipeline("230.1.1.1", 1720, "enP8p1s0", "avdec_h264")
    assert "udpsrc address=230.1.1.1 port=1720 multicast-iface=enP8p1s0" in desc
    assert "encoding-name=H264" in desc and "appsink name=sink" in desc
    assert "multicast-iface" not in build_pipeline("230.1.1.1", 1720, "", "avdec_h264")


def test_decodes_stream_and_recovers_after_restart_with_new_resolution():
    node = FrontCameraNode(
        parameter_overrides=[
            Parameter("address", value="127.0.0.1"),
            Parameter("port", value=PORT),
            Parameter("restart_after_s", value=1.0),
            Parameter("latency_s", value=0.1),
        ]
    )
    io = rclpy.create_node("front_camera_test_io")
    frames = []
    io.create_subscription(Image, "/camera/front/image_raw", frames.append, qos_profile_sensor_data)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(io)

    def spin_until(cond, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not cond():
            ex.spin_once(timeout_sec=0.02)
        return cond()

    sender = _sender(320, 240)
    try:
        assert spin_until(lambda: len(frames) >= 10, 10.0), f"got {len(frames)} frames"
        f = frames[-1]
        assert (f.width, f.height, f.encoding, f.header.frame_id) == (320, 240, "bgr8", "front_camera_optical_frame")
        age = (io.get_clock().now() - rclpy.time.Time.from_msg(f.header.stamp)).nanoseconds / 1e9
        assert 0.1 <= age < 1.0  # stamped at receipt minus latency_s

        # Stream stops (robot video service restart), then returns with a new SPS.
        _stop(sender)
        spin_until(lambda: node.restarts >= 1, 5.0)
        assert node.restarts >= 1
        sender = _sender(640, 480)
        assert spin_until(lambda: frames[-1].width == 640, 10.0)
        assert frames[-1].height == 480
    finally:
        if sender.poll() is None:
            _stop(sender)
        ex.shutdown()
        node.destroy_node()
        io.destroy_node()
