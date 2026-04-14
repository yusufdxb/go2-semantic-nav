"""slam_toolbox fallback launch — for when the sibling seeing-eye-dog stack's
`/odom` is stuck or unavailable. Brings up slam_toolbox in async mode so semantic-nav
has a stable `map → odom → base_link` TF chain even when no external SLAM is running.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("odom_frame", default_value="odom"),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("log_level", default_value="info"),
    ]

    slam_params = {
        "use_sim_time": LaunchConfiguration("use_sim_time"),
        "base_frame": LaunchConfiguration("base_frame"),
        "odom_frame": LaunchConfiguration("odom_frame"),
        "map_frame": LaunchConfiguration("map_frame"),
        "scan_topic": LaunchConfiguration("scan_topic"),
        "mode": "mapping",
        "resolution": 0.05,
        "max_laser_range": 8.0,
        "minimum_time_interval": 0.25,
        "transform_publish_period": 0.05,
        "map_update_interval": 2.0,
        "do_loop_closing": True,
        "loop_search_maximum_distance": 3.0,
        "scan_buffer_size": 10,
    }

    slam_toolbox = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        emulate_tty=True,
        parameters=[slam_params],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription(args + [slam_toolbox])
