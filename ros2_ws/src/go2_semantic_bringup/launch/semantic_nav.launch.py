"""Top-level launch for go2-semantic-nav: detector + scene_graph + grounding + RViz preset.

This launch does NOT start RealSense, Nav2, or the GO2-seeing-eye-dog stack. Those must
already be running (via `ros2 launch go2_bringup go2_full.launch.py ...`) or be replayed
from a rosbag before this launch is invoked.

Args:
  device            : torch device string, default "cuda:0"
  backend           : detector backend id (e.g., "yolo_world_v2_s", "yoloe_11s")
  segmenter         : segmenter backend id ("mobile_sam", "nano_sam", ...)
  encoder           : CLIP backend id ("openclip_vit_b16", "mobileclip_s2", ...)
  prompt_classes_file : optional YAML with open-vocab prompts (override)
  detection_rate_hz : detector timer rate
  publish_rate_hz   : scene-graph publish rate
  enable_detector   : set false to use an offboard detector
  enable_scene_graph: set false to skip scene graph (debug)
  enable_grounding  : set false to skip grounding action server
  use_rviz          : launch RViz with the provided preset
  log_level         : default info
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    detector_share = get_package_share_directory("go2_open_vocab_detector")
    scene_graph_share = get_package_share_directory("go2_scene_graph")
    grounding_share = get_package_share_directory("go2_language_grounding")
    bringup_share = get_package_share_directory("go2_semantic_bringup")

    default_detector_yaml = os.path.join(detector_share, "config", "detector.yaml")
    default_scene_graph_yaml = os.path.join(scene_graph_share, "config", "scene_graph.yaml")
    default_grounding_yaml = os.path.join(grounding_share, "config", "grounding.yaml")
    default_rviz = os.path.join(bringup_share, "rviz", "semantic_nav.rviz")

    args = [
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("backend", default_value="yolo_world_v2_s"),
        DeclareLaunchArgument("segmenter", default_value="mobile_sam"),
        DeclareLaunchArgument("encoder", default_value="openclip_vit_b16"),
        DeclareLaunchArgument("prompt_classes_file", default_value=""),
        DeclareLaunchArgument("detection_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("publish_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("enable_detector", default_value="true"),
        DeclareLaunchArgument("enable_scene_graph", default_value="true"),
        DeclareLaunchArgument("enable_grounding", default_value="true"),
        DeclareLaunchArgument("use_slam_fallback", default_value="false",
                              description="If true, launch slam_toolbox to produce map→odom TF when /odom is unavailable"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),
        DeclareLaunchArgument("detector_params", default_value=default_detector_yaml),
        DeclareLaunchArgument("scene_graph_params", default_value=default_scene_graph_yaml),
        DeclareLaunchArgument("grounding_params", default_value=default_grounding_yaml),
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="Set on all nodes; pair with `ros2 bag play --clock` when replaying a bag"),
        DeclareLaunchArgument("log_level", default_value="info"),
    ]

    detector_node = Node(
        package="go2_open_vocab_detector",
        executable="detector_node",
        name="go2_open_vocab_detector",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("detector_params"),
            {
                "device": LaunchConfiguration("device"),
                "detector_backend": LaunchConfiguration("backend"),
                "segmenter_backend": LaunchConfiguration("segmenter"),
                "encoder_backend": LaunchConfiguration("encoder"),
                "detection_rate_hz": LaunchConfiguration("detection_rate_hz"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        condition=IfCondition(LaunchConfiguration("enable_detector")),
    )

    scene_graph_node = Node(
        package="go2_scene_graph",
        executable="scene_graph_node",
        name="go2_scene_graph",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("scene_graph_params"),
            {
                "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        condition=IfCondition(LaunchConfiguration("enable_scene_graph")),
    )

    grounding_node = Node(
        package="go2_language_grounding",
        executable="grounding_node",
        name="go2_language_grounding",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("grounding_params"),
            {
                "device": LaunchConfiguration("device"),
                "encoder_backend": LaunchConfiguration("encoder"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        condition=IfCondition(LaunchConfiguration("enable_grounding")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_semantic_nav",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    slam_fallback = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[{
            "base_frame": "base_link",
            "odom_frame": "odom",
            "map_frame": "map",
            "scan_topic": "/scan",
            "mode": "mapping",
            "resolution": 0.05,
            "max_laser_range": 8.0,
            "transform_publish_period": 0.05,
        }],
        condition=IfCondition(LaunchConfiguration("use_slam_fallback")),
    )

    banner = LogInfo(
        msg=[
            "go2-semantic-nav bringup: backend=",
            LaunchConfiguration("backend"),
            ", seg=",
            LaunchConfiguration("segmenter"),
            ", enc=",
            LaunchConfiguration("encoder"),
            ", device=",
            LaunchConfiguration("device"),
        ]
    )

    return LaunchDescription(
        args + [banner, slam_fallback, detector_node, scene_graph_node, grounding_node, rviz]
    )
