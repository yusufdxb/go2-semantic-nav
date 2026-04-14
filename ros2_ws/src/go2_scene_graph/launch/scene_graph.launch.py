"""Launch go2_scene_graph standalone."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("go2_scene_graph")
    default_params = os.path.join(pkg_share, "config", "scene_graph.yaml")

    params_arg = DeclareLaunchArgument("params_file", default_value=default_params)
    log_level_arg = DeclareLaunchArgument("log_level", default_value="info")

    node = Node(
        package="go2_scene_graph",
        executable="scene_graph_node",
        name="go2_scene_graph",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("params_file")],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )
    return LaunchDescription([params_arg, log_level_arg, node])
