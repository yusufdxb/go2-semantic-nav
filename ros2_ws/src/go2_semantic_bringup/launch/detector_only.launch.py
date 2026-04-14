"""Launch only the detector + RViz, for quick visual validation on a rosbag."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    detector_share = get_package_share_directory("go2_open_vocab_detector")
    bringup_share = get_package_share_directory("go2_semantic_bringup")
    default_params = os.path.join(detector_share, "config", "detector.yaml")
    default_rviz = os.path.join(bringup_share, "rviz", "semantic_nav.rviz")

    args = [
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),
        DeclareLaunchArgument("log_level", default_value="info"),
    ]

    detector = Node(
        package="go2_open_vocab_detector",
        executable="detector_node",
        name="go2_open_vocab_detector",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("params_file"),
            {"device": LaunchConfiguration("device")},
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_detector_only",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription(args + [detector, rviz])
