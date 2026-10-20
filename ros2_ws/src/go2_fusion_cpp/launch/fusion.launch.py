"""Isolated fusion adapter; starts no drivers, detector, or motion nodes."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = str(
        Path(get_package_share_directory("go2_fusion_cpp")) / "config" / "fusion.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="go2_fusion_cpp",
                executable="fusion_node",
                name="go2_fusion",
                parameters=[
                    LaunchConfiguration("config"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
                output="screen",
            ),
        ]
    )
