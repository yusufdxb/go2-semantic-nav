"""Launch semantic mapping, optionally using validated fusion positions."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _bool(value, name):
    if value.lower() not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return value.lower() == "true"


def _nodes(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    enabled = _bool(value("fusion_mapping"), "fusion_mapping")
    overrides = {}
    for name in ("map_frame", "detections_topic", "snapshot_dir"):
        if value(name):
            overrides[name] = value(name)
    if value("use_sim_time"):
        overrides["use_sim_time"] = _bool(value("use_sim_time"), "use_sim_time")
    adapter_parameters = None
    if enabled:
        # Resolve the mapper's configured values so its optional adapter uses the
        # same frame and semantic input. Defaults do not override custom YAML.
        with open(value("params_file"), encoding="utf-8") as source:
            config = yaml.safe_load(source) or {}
        configured = dict(config.get("/**", {}).get("ros__parameters", {}))
        for node_name in ("go2_scene_graph", "/go2_scene_graph"):
            configured.update(config.get(node_name, {}).get("ros__parameters", {}))
        configured.update(overrides)
        semantic_topic = configured.get("detections_topic", "/semantic/detections")
        map_frame = configured.get("map_frame", "map")
        sim_time = configured.get("use_sim_time", False)
        adapter_parameters = {
            "map_frame": map_frame,
            "use_sim_time": sim_time,
            "semantic_topic": semantic_topic,
            "fusion_topic": value("fusion_topic"),
            "output_topic": value("fused_detections_topic"),
        }
        overrides["detections_topic"] = value("fused_detections_topic")
    nodes = [
        Node(
            package="go2_scene_graph",
            executable="scene_graph_node",
            name="go2_scene_graph",
            output="screen",
            emulate_tty=True,
            parameters=[value("params_file"), overrides],
            arguments=["--ros-args", "--log-level", value("log_level")],
        )
    ]
    if enabled:
        nodes.append(
            Node(
                package="go2_scene_graph",
                executable="fusion_semantic_adapter",
                name="fusion_semantic_adapter",
                output="screen",
                parameters=[adapter_parameters],
                arguments=["--ros-args", "--log-level", value("log_level")],
            )
        )
    return nodes


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("go2_scene_graph")
    default_params = os.path.join(pkg_share, "config", "scene_graph.yaml")
    defaults = {
        "params_file": default_params,
        "log_level": "info",
        "fusion_mapping": "false",
        "detections_topic": "",
        "fusion_topic": "/fusion/objects",
        "fused_detections_topic": "/semantic/fused_detections",
        "map_frame": "",
        "use_sim_time": "",
        "snapshot_dir": "",
    }
    return LaunchDescription(
        [
            *(
                DeclareLaunchArgument(name, default_value=default)
                for name, default in defaults.items()
            ),
            OpaqueFunction(function=_nodes),
        ]
    )
