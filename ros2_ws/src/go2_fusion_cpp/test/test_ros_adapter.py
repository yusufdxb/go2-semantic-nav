#!/usr/bin/env python3
"""Synthetic isolated ROS adapter safety tests. No hardware or recording mutation."""

import json
import math
import os
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", required=True, type=Path, help="Installed fusion_node executable"
    )
    parser.add_argument("--results", type=Path, help="Optional JSON result artifact")
    parser.add_argument(
        "--domain-id", type=int, default=190, help="Isolated local ROS domain (0..232)"
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "cuda_mapped"), default="cpu")
    options = parser.parse_args()
    if not options.executable.is_file():
        parser.error("Executable does not exist")
    if not 0 <= options.domain_id <= 232:
        parser.error("Domain ID must be in [0,232]")
    os.environ["ROS_DOMAIN_ID"] = str(options.domain_id)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
    from geometry_msgs.msg import TransformStamped
    from go2_semantic_msgs.msg import SemanticDetection, SemanticDetectionArray
    from go2_fusion_cpp.msg import FusionArray
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from tf2_ros.transform_broadcaster import TransformBroadcaster
    from builtin_interfaces.msg import Time
    from rclpy.parameter import Parameter
    from rcl_interfaces.srv import SetParameters, GetParameters

    EXE = str(options.executable.resolve())
    TEMP = tempfile.TemporaryDirectory(prefix="fusion_synthetic_")
    p = Path(TEMP.name)
    (p / "intrinsics.yaml").write_text(
        "nominal: false\nimage_width: 32\nimage_height: 24\ndistortion_model: plumb_bob\ncamera_matrix:\n  data: [20, 0, 16, 0, 20, 12, 0, 0, 1]\ndistortion_coefficients:\n  data: [0, 0, 0, 0, 0]\n"
    )
    (p / "extrinsics.yaml").write_text(
        "nominal: false\nparent_frame: synthetic_base\nchild_frame: synthetic_camera\nxyz: [0, 0, 0]\nrpy: [0, 0, 0]\n"
    )
    results = []
    rclpy.init()
    node = rclpy.create_node("fusion_adapter_synthetic_test")
    qos = QoSProfile(depth=32, reliability=ReliabilityPolicy.RELIABLE)
    pubs = [
        node.create_publisher(Image, "/fusion/test/image", qos),
        node.create_publisher(CameraInfo, "/fusion/test/info", qos),
        node.create_publisher(PointCloud2, "/fusion/test/cloud", qos),
        node.create_publisher(SemanticDetectionArray, "/fusion/test/detections", qos),
    ]
    received = []
    sub = node.create_subscription(
        FusionArray, "/fusion/test/objects", lambda m: received.append(m), qos
    )
    broadcaster = StaticTransformBroadcaster(node)
    proc = None
    last_log = None
    node_name = ""
    generation = 0

    def spin(sec):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=min(0.02, max(0, end - time.monotonic())))

    def start(trusted=False, timing=True, cloud_geometry=True, sim=False):
        nonlocal proc, last_log, node_name, generation
        if proc:
            proc.terminate()
            proc.wait(timeout=5)
        received.clear()
        generation += 1
        node_name = f"fusion_adapter_under_test_{generation}"
        params = {
            "input_type": "semantic",
            "backend": options.backend,
            "image_topic": "/fusion/test/image",
            "camera_info_topic": "/fusion/test/info",
            "cloud_topic": "/fusion/test/cloud",
            "semantic_detections_topic": "/fusion/test/detections",
            "output_topic": "/fusion/test/objects",
            "fixed_frame": "synthetic_odom",
            "camera_frame": "synthetic_camera",
            "max_width": 32,
            "max_height": 24,
            "calibration_validated": trusted,
            "timing_validated": timing,
            "cloud_geometry_validated": cloud_geometry,
            "acquisition_clock_validated": True,
            "calibration_id": "synthetic-fixture-only",
            "intrinsics_file": str(p / "intrinsics.yaml"),
            "extrinsics_file": str(p / "extrinsics.yaml"),
            "pending_timeout_ms": 50.0,
            "use_sim_time": sim,
        }
        args = [EXE, "--ros-args", "-r", "__node:=" + node_name]
        for key, val in params.items():
            args += [
                "-p",
                key + ":=" + str(val).lower() if isinstance(val, bool) else key + ":=" + str(val),
            ]
        last_log = p / f"node_{time.monotonic_ns()}.log"
        with last_log.open("w") as log:
            proc = subprocess.Popen(args, stdout=log, stderr=log)
        spin(0.4)
        end = time.monotonic() + 4
        while any(pub.get_subscription_count() < 1 for pub in pubs) and time.monotonic() < end:
            spin(0.1)
        assert proc.poll() is None, "node failed to start"
        assert all(
            pub.get_subscription_count() >= 1 for pub in pubs
        ), "DDS publisher matching failed"
        startup_client = node.create_client(GetParameters, f"/{node_name}/get_parameters")
        assert startup_client.wait_for_service(timeout_sec=3)
        request = startup_client.call_async(GetParameters.Request(names=["backend"]))
        deadline = time.monotonic() + 5
        while not request.done() and time.monotonic() < deadline:
            spin(0.02)
        assert request.done(), "New node did not complete startup service handshake"
        assert request.result().values[0].string_value == options.backend
        node.destroy_client(startup_client)
        spin(0.15)

    def stamp(t):
        return Time(sec=t, nanosec=0)

    def inputs(t):
        image = Image()
        image.header.frame_id = "synthetic_camera"
        image.header.stamp = stamp(t)
        image.width = 32
        image.height = 24
        image.encoding = "rgb8"
        image.step = 96
        image.data = [0] * (24 * 96)
        info = CameraInfo()
        info.header = image.header
        info.width = 32
        info.height = 24
        info.distortion_model = "plumb_bob"
        info.k = [20.0, 0.0, 16.0, 0.0, 20.0, 12.0, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        cloud = PointCloud2()
        cloud.header.frame_id = "synthetic_odom"
        cloud.header.stamp = stamp(t)
        cloud.height = 1
        cloud.width = 49
        cloud.point_step = 12
        cloud.row_step = 49 * 12
        cloud.fields = [
            PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate(("x", "y", "z"))
        ]
        cloud.data = b"".join(
            struct.pack("<fff", (x - 3) * 0.12, (y - 3) * 0.12, 2.0)
            for y in range(7)
            for x in range(7)
        )
        detections = SemanticDetectionArray()
        detections.header = image.header
        detections.source_frame_id = "synthetic_camera"
        d = SemanticDetection()
        d.label = "chair"
        d.score = 0.9
        d.bbox_2d.center.position.x = 16.0
        d.bbox_2d.center.position.y = 12.0
        d.bbox_2d.size_x = 24.0
        d.bbox_2d.size_y = 20.0
        detections.detections = [d]
        return [image, info, cloud, detections]

    def tf(offset=0.0):
        msgs = []
        for parent, child, x in [
            ("synthetic_odom", "synthetic_base", 0.0),
            ("synthetic_base", "synthetic_camera", offset),
        ]:
            m = TransformStamped()
            m.header.frame_id = parent
            m.child_frame_id = child
            m.header.stamp = node.get_clock().now().to_msg()
            m.transform.rotation.w = 1.0
            m.transform.translation.x = x
            msgs.append(m)
        broadcaster.sendTransform(msgs)
        spin(0.15)

    def run(name, data, status, send=(0, 1, 2, 3)):
        previous = len(received)
        for i in send:
            if i != 3:
                pubs[i].publish(data[i])
        if 3 in send:
            if len(send) > 1:
                spin(0.03)
            pubs[3].publish(data[3])
        end = time.monotonic() + 2
        while len(received) == previous and time.monotonic() < end:
            spin(0.01)
        assert len(received) > previous, (name, "no result")
        out = received[-1]
        assert out.status == status, (name, out.status, out.status_detail)
        if status == 0:
            assert out.fusion_compute_ms >= 0
        elif name in (
            "nominal_default_gate",
            "timing_unvalidated_gate",
            "cloud_geometry_unvalidated_gate",
            "missing_tf",
            "stale_dynamic_tf",
            "dropped_source_image",
        ):
            assert out.fusion_compute_ms == -1
        if status != 0:
            assert all(
                not o.valid
                and o.track_id == 0
                and math.isnan(o.position.x)
                and all(math.isnan(v) for v in o.covariance)
                for o in out.objects
            ), (name, "invalid result leaked geometry")
        results.append(
            {
                "case": name,
                "status": out.status,
                "detail": out.status_detail,
                "fusion_compute_ms": out.fusion_compute_ms,
                "sensor_latency_valid": out.sensor_latency_valid,
            }
        )
        return out

    try:
        start(False)
        run("nominal_default_gate", inputs(100), 1, send=(3,))
        start(True, timing=False)
        run("timing_unvalidated_gate", inputs(101), 1, send=(3,))
        start(True, cloud_geometry=False)
        run("cloud_geometry_unvalidated_gate", inputs(101), 1, send=(3,))
        start(True)
        run("missing_tf", inputs(102), 2)
        tf()
        out = run("calibrated_synthetic_positive", inputs(103), 0)
        assert (
            len(out.objects) == 1
            and out.objects[0].valid
            and out.objects[0].track_id > 0
            and out.objects[0].support_points >= 6
        )
        assert abs(out.objects[0].position.z - 2) < 1e-6
        data = inputs(103)
        for message in data:
            message.header.stamp.nanosec = 200000000
        out2 = run("track_continuity", data, 0)
        assert out2.objects[0].track_id == out.objects[0].track_id
        data = inputs(105)
        data[2].header.stamp.nanosec = 70000000
        run("timestamp_skew_70ms", data, 3)
        run("dropped_source_image", inputs(106), 3, send=(1, 2, 3))
        data = inputs(107)
        data[2].fields = data[2].fields[:2]
        run("malformed_point_fields", data, 4)
        data = inputs(108)
        data[2].point_step = 4
        run("malformed_point_stride", data, 4)
        data = inputs(109)
        data[0].data = []
        run("malformed_image_payload", data, 4)
        data = inputs(110)
        data[1].k[0] = 21
        run("camera_info_calibration_mismatch", data, 1)
        tf(0.02)
        run("tf_calibration_mismatch", inputs(111), 1)
        tf()
        data = inputs(112)
        d = data[3].detections[0]
        d.mask_roi.width = 32
        d.mask_roi.height = 24
        d.mask_rle = [255, 0, 255, 0, 255, 0, 4]
        run("malformed_rle_overrun", data, 4)
        data = inputs(113)
        d = data[3].detections[0]
        d.mask_roi.width = 32
        d.mask_roi.height = 24
        d.mask_rle = [0, 255, 0, 255, 0, 255, 0, 3]
        out = run("chunked_rle_positive", data, 0)
        assert out.objects[0].valid
        run("out_of_order_frame", inputs(104), 8)
        dynamic = TransformBroadcaster(node)
        stale = TransformStamped()
        stale.header.frame_id = "synthetic_odom"
        stale.header.stamp = stamp(114)
        stale.child_frame_id = "synthetic_lidar"
        stale.transform.rotation.w = 1.0
        dynamic.sendTransform(stale)
        spin(0.15)
        data = inputs(115)
        data[2].header.frame_id = "synthetic_lidar"
        run("stale_dynamic_tf", data, 2)
        prev = len(received)
        for t in (114, 115, 116):
            pubs[3].publish(inputs(t)[3])
        spin(0.2)
        outs = received[prev:]
        assert len(outs) == 3 and any(
            o.status == 5 and o.dropped_pending_batches > 0 for o in outs
        ), [o.status for o in outs]
        results.append({"case": "pending_queue_overflow", "statuses": [o.status for o in outs]})
        # Reset process into simulated clock, ensure fake sensor latency is never advertised.
        start(True, sim=True)
        tf()
        out = run("sim_clock_latency_invalid", inputs(117), 0)
        assert not out.sensor_latency_valid and out.sensor_to_output_ms == -1
        client = node.create_client(SetParameters, f"/{node_name}/set_parameters")
        assert client.wait_for_service(timeout_sec=2)
        future = client.call_async(
            SetParameters.Request(
                parameters=[Parameter("calibration_validated", value=False).to_parameter_msg()]
            )
        )
        deadline = time.monotonic() + 2
        while not future.done() and time.monotonic() < deadline:
            spin(0.01)
        assert future.done() and not future.result().results[0].successful
        results.append({"case": "runtime_calibration_mutation_rejected"})

        def resident_kib():
            for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
            raise AssertionError("No process resident memory measurement")

        rss_before = resident_kib()
        rss_peak = rss_before
        for frame in range(100):
            start_time = time.monotonic()
            data = inputs(120 + frame // 20)
            for message in data:
                message.header.stamp.nanosec = (frame % 20) * 50_000_000
            output = run("sustained_frame", data, 0)
            assert output.objects[0].valid
            results.pop()
            rss_peak = max(rss_peak, resident_kib())
            spin(max(0, 0.05 - (time.monotonic() - start_time)))
        rss_after = resident_kib()
        assert rss_after - rss_before < 8192, (rss_before, rss_after)
        results.append(
            {
                "case": "sustained_20hz_synthetic_100_frames",
                "rss_before_kib": rss_before,
                "rss_after_kib": rss_after,
                "rss_peak_kib": rss_peak,
            }
        )
        if options.results:
            options.results.parent.mkdir(parents=True, exist_ok=True)
            options.results.write_text(
                json.dumps(
                    {
                        "synthetic_only": True,
                        "backend": options.backend,
                        "passed": len(results),
                        "cases": results,
                    },
                    indent=2,
                )
                + "\n"
            )
        print(
            json.dumps(
                {
                    "passed": len(results),
                    "results": str(options.results) if options.results else None,
                }
            )
        )
    except Exception:
        if last_log:
            print(last_log.read_text(), flush=True)
        raise
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown()
        TEMP.cleanup()


if __name__ == "__main__":
    main()
