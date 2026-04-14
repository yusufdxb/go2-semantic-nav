"""Launch go2_open_vocab_detector in standalone mode (no scene graph, no grounding)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("go2_open_vocab_detector")
    default_params = os.path.join(pkg_share, "config", "detector.yaml")

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="YAML params file for the detector node.",
    )
    log_level_arg = DeclareLaunchArgument("log_level", default_value="info")

    detector = Node(
        package="go2_open_vocab_detector",
        executable="detector_node",
        name="go2_open_vocab_detector",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("params_file")],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription([params_arg, log_level_arg, detector])
