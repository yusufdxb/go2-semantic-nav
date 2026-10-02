#!/usr/bin/env python3
"""Rehearsal only: stand-in for the GO2's own odometry and deskewed LiDAR.

Publishes /utlidar/robot_odom (150 Hz) and /utlidar/cloud_deskewed (15 Hz, in
odom) stamped on a robot clock skewed by the measured -27,605,481 s, so the
base stack relay's clock-offset path is exercised. The base yaws slowly back
and forth; the cloud is a wall 3 m ahead with a box 1.8 m ahead.
"""
import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

SKEW_NS = -27_605_481 * 1_000_000_000


def main() -> None:
    rclpy.init()
    n = rclpy.create_node("fake_go2_sensors")
    odom_pub = n.create_publisher(Odometry, "/utlidar/robot_odom", qos_profile_sensor_data)
    cloud_pub = n.create_publisher(PointCloud2, "/utlidar/cloud_deskewed", qos_profile_sensor_data)
    wall = [[3.0, y, z] for y in np.arange(-2.5, 2.5, 0.05) for z in np.arange(-0.4, 1.2, 0.05)]
    box = [[1.8, y, z] for y in np.arange(-0.3, 0.3, 0.03) for z in np.arange(-0.3, 0.3, 0.03)]
    pts = np.array(wall + box, np.float32)
    t0 = time.monotonic()
    k = 0

    def stamp():
        t = n.get_clock().now().nanoseconds + SKEW_NS
        h = Header()
        h.stamp.sec, h.stamp.nanosec = divmod(t, 1_000_000_000)
        return h

    while rclpy.ok():
        t = time.monotonic() - t0
        yaw = 0.3 * math.sin(0.5 * t)
        od = Odometry(header=stamp(), child_frame_id="base_link")
        od.header.frame_id = "odom"
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        od.twist.twist.angular.z = 0.15 * math.cos(0.5 * t)
        odom_pub.publish(od)
        if k % 10 == 0:
            h = stamp()
            h.frame_id = "odom"
            h.stamp.nanosec = max(0, h.stamp.nanosec - 30_000_000)  # sweep ends just before the newest odom
            cloud_pub.publish(point_cloud2.create_cloud_xyz32(h, pts))
        k += 1
        rclpy.spin_once(n, timeout_sec=0.0)
        time.sleep(1 / 150)


if __name__ == "__main__":
    main()
