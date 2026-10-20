"""front_camera_node: GO2 front camera (RTP H.264 multicast) -> sensor_msgs/Image.

The GO2 pushes its front camera as RTP H.264 to multicast 230.1.1.1:1720 on the
robot's 192.168.123.0/24 segment. This node decodes it with GStreamer and
publishes BGR frames stamped with the LOCAL receive time minus
``latency_s`` (the robot's own clock is skewed by months, so no robot stamp is
used), in ``frame_id`` = the calibrated camera optical frame.

A decoder that stops emitting after the stream restarts (new SPS) looks
identical to a camera that went away, so if no frame arrives for
``restart_after_s`` the pipeline is torn down and rebuilt.

Receive-only: joining a multicast group sends nothing to the robot and does
not disturb other listeners.

GStreamer is imported only after the ROS node exists. libgstreamer pulls in
libunwind, whose _Unwind_Resume interposes on libgcc_s's; Fast DDS throws and
catches an exception while opening its shared-memory transport during node
creation, and unwinding it through libunwind aborts the process. Creating the
node first binds the symbol to libgcc_s. (CycloneDDS is unaffected.)
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .image_msg import fill_image


def _import_gst():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst

DEFAULT_PIPELINE = (
    "udpsrc address={address} port={port}{iface} "
    "caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264\" "
    "! rtph264depay ! h264parse ! {decoder} ! videoconvert ! video/x-raw,format=BGR "
    "! appsink name=sink drop=true max-buffers=1 sync=false"
)


def build_pipeline(address: str, port: int, iface: str, decoder: str) -> str:
    return DEFAULT_PIPELINE.format(
        address=address,
        port=int(port),
        iface=f" multicast-iface={iface}" if iface else "",
        decoder=decoder,
    )


class FrontCameraNode(Node):
    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__("go2_front_camera", parameter_overrides=parameter_overrides or [])
        p = self.declare_parameter
        p("address", "230.1.1.1")
        p("port", 1720)
        p("multicast_iface", "")  # e.g. enP8p1s0 on the GO2 Jetson; empty = OS default route
        p("decoder", "avdec_h264")
        p("pipeline", "")  # full override; must end in an appsink named "sink" producing BGR
        p("image_topic", "/camera/front/image_raw")
        p("frame_id", "front_camera_optical_frame")
        p("latency_s", 0.0)  # capture-to-receive delay subtracted from the stamp
        p("max_rate_hz", 15.0)
        p("restart_after_s", 3.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self._desc = str(g("pipeline")) or build_pipeline(
            str(g("address")), int(g("port")), str(g("multicast_iface")), str(g("decoder"))
        )
        self._frame_id = str(g("frame_id"))
        self._latency = Duration(nanoseconds=int(float(g("latency_s")) * 1e9))
        self._min_period = 1.0 / float(g("max_rate_hz")) if float(g("max_rate_hz")) > 0 else 0.0
        self._restart_after = float(g("restart_after_s"))
        self._pub = self.create_publisher(Image, str(g("image_topic")), qos_profile_sensor_data)

        self._gst = _import_gst()  # after super().__init__: see module docstring
        self.frames = 0
        self.restarts = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="front_camera_gst", daemon=True)
        self._thread.start()
        self.get_logger().info(f"front camera pipeline: {self._desc}")

    def _run(self) -> None:
        while not self._stop.is_set():
            Gst = self._gst
            pipeline = Gst.parse_launch(self._desc)
            sink = pipeline.get_by_name("sink")
            pipeline.set_state(Gst.State.PLAYING)
            last_frame = time.monotonic()
            last_pub = 0.0
            try:
                while not self._stop.is_set():
                    sample = sink.emit("try-pull-sample", int(0.2 * Gst.SECOND))
                    now = time.monotonic()
                    if sample is None:
                        if now - last_frame > self._restart_after:
                            self.get_logger().warn(
                                f"no camera frame for {self._restart_after:.1f} s; rebuilding pipeline",
                                throttle_duration_sec=10.0)
                            self.restarts += 1
                            break
                        continue
                    last_frame = now
                    if now - last_pub < self._min_period:
                        continue
                    last_pub = now
                    self._publish(sample)
            finally:
                pipeline.set_state(Gst.State.NULL)

    def _publish(self, sample) -> None:
        stamp = self.get_clock().now() - self._latency
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        buf = sample.get_buffer()
        ok, info = buf.map(self._gst.MapFlags.READ)
        if not ok:
            return
        try:
            stride = info.size // height
            frame = np.frombuffer(info.data, np.uint8).reshape(height, stride)[:, : width * 3]
            frame = frame.reshape(height, width, 3)
            msg = fill_image(Image(), frame, "bgr8")
        finally:
            buf.unmap(info)
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self._frame_id
        self._pub.publish(msg)
        self.frames += 1

    def destroy_node(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrontCameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
