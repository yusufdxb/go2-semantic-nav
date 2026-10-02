"""
deploy.launch.py: semantic navigation on the GO2 motion-authority stack.

    ros2 launch go2_semantic_bringup deploy.launch.py \\
        planner:=nav2 localization:=slam_mapping hardware_adapter:=dry_run

Brings up, in one graph:

* go2_bringup/system.launch.py (from GO2-seeing-eye-dog, built in an
  underlay): localization (odom/LiDAR relay, pointcloud_to_laserscan,
  slam_toolbox), the planner (Nav2, or the staged approach controller behind
  a NavigateToPose adapter), the LiDAR hazard source, and the motion
  authority (safety arbiter -> hardware bridge).
* go2_rgb_lidar/rgb_lidar.launch.py (camera:=rgb_lidar, the default): the
  GO2 front camera driver and LiDAR-projected depth aligned to it. The GO2
  has no depth camera; without this the detector never fires.
* semantic_nav.launch.py: detector, scene graph and grounding. Grounding
  dispatches NavigateToPose to whichever planner is running, so it never
  talks to the robot directly; every velocity passes the arbiter.

Motion requires ALL of: hardware_adapter:=unitree_sport, allow_goal_publication:=true,
and a GroundAndNavigate request with dry_run: false. The defaults are the
safe ones (dry_run adapter, interlock off).

Arguments
---------
planner                 nav2 | staged_nav   (default nav2). staged_nav is the
                        straight-line approach controller behind a
                        NavigateToPose adapter: no planning, no obstacle
                        avoidance, no costmap (the grounding costmap gate is
                        switched off with it).
localization            slam_mapping | slam_localization   (default slam_mapping)
map_file                serialized slam_toolbox pose graph for slam_localization
hardware_adapter        dry_run | unitree_sport   (default dry_run)
allow_goal_publication  semantic interlock, default false
camera                  rgb_lidar | rgbd   (default rgb_lidar). rgb_lidar = GO2
                        front camera + LiDAR depth; rgbd = an external RGB-D
                        camera publishing aligned depth (set the three topics)
camera_intrinsics_file  rgb_lidar calibration (default: shipped NOMINAL file)
camera_extrinsics_file  rgb_lidar camera pose in base_link (default: NOMINAL)
allow_nominal_calibration  rgb_lidar: false withholds depth until calibrated
publish_lidar_overlay   rgb_lidar: publish /camera/front/lidar_overlay
multicast_iface         rgb_lidar: robot-side NIC for the camera multicast
camera_impl             rgb_lidar: cpp (default) | py camera driver
camera_decoder          rgb_lidar + cpp: nvv4l2 (default, Jetson) | avdec
enable_detector         false to run without object detection (default true)
enable_scene_graph      false when another node provides /semantic/scene_graph
cloud_in_topic          /utlidar/cloud_deskewed (default) or /utlidar/cloud
publish_lidar_extrinsic true only with the raw sensor-frame cloud
image_topic, depth_topic, camera_info_topic: empty = the camera mode's
                        topics; detector_params: empty = the camera mode's file
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare

# Per camera mode: (image, depth, camera_info, detector params file).
_CAMERA_MODES = {
    "rgb_lidar": ("/camera/front/image_raw", "/camera/front/lidar_depth", "/camera/front/camera_info",
                  "detector_rgb_lidar.yaml"),
    "rgbd": ("/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", "/camera/color/camera_info",
             "detector.yaml"),
}


def _per_mode(cfg, override: str, index: int):
    """The override if set, else the camera mode's value (unknown mode -> launch error)."""
    table = {mode: values[index] for mode, values in _CAMERA_MODES.items()}
    return PythonExpression(
        ["'", cfg[override], "' or ", repr(table), "['", cfg["camera"], "']"]
    )

_ARGS = {
    "planner": "nav2",
    "localization": "slam_mapping",
    "map_file": "",
    "hardware_adapter": "dry_run",
    "dry_run_log_path": "",
    "allow_goal_publication": "false",
    "camera": "rgb_lidar",
    "allow_nominal_calibration": "false",
    "publish_lidar_overlay": "false",
    "multicast_iface": "",
    "camera_impl": "cpp",
    "camera_decoder": "nvv4l2",
    "detector_params": "",
    "enable_detector": "true",
    "enable_scene_graph": "true",
    "cloud_in_topic": "/utlidar/cloud_deskewed",
    "publish_lidar_extrinsic": "false",
    "image_topic": "",
    "depth_topic": "",
    "camera_info_topic": "",
    "use_sim_time": "false",
    "log_level": "info",
}


def generate_launch_description() -> LaunchDescription:
    cfg = {name: LaunchConfiguration(name) for name in _ARGS}
    for name in ("camera_intrinsics_file", "camera_extrinsics_file"):
        cfg[name] = LaunchConfiguration(name)
    rgb_lidar_share = FindPackageShare("go2_rgb_lidar")
    calibration_args = [
        DeclareLaunchArgument(
            "camera_intrinsics_file",
            default_value=PathJoinSubstitution([rgb_lidar_share, "config", "front_camera_intrinsics_nominal.yaml"]),
        ),
        DeclareLaunchArgument(
            "camera_extrinsics_file",
            default_value=PathJoinSubstitution([rgb_lidar_share, "config", "front_camera_extrinsics_nominal.yaml"]),
        ),
    ]
    detector_params = PythonExpression(
        ["'", cfg["detector_params"], "' or '", FindPackageShare("go2_open_vocab_detector"), "/config/' + ",
         repr({m: v[3] for m, v in _CAMERA_MODES.items()}), "['", cfg["camera"], "']"]
    )

    rgb_lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([rgb_lidar_share, "launch", "rgb_lidar.launch.py"])]
        ),
        launch_arguments={
            "intrinsics_file": cfg["camera_intrinsics_file"],
            "extrinsics_file": cfg["camera_extrinsics_file"],
            "allow_nominal_calibration": cfg["allow_nominal_calibration"],
            "publish_overlay": cfg["publish_lidar_overlay"],
            "multicast_iface": cfg["multicast_iface"],
            "camera_impl": cfg["camera_impl"],
            "camera_decoder": cfg["camera_decoder"],
            "use_sim_time": cfg["use_sim_time"],
        }.items(),
        condition=IfCondition(PythonExpression(["'", cfg["camera"], "' == 'rgb_lidar'"])),
    )

    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("go2_bringup"), "launch", "system.launch.py"])]
        ),
        launch_arguments={
            # The seeing-eye-dog perception (mic, YOLO person detector, depth
            # safety monitor) is not part of this deployment; LiDAR provides
            # the hazard context instead.
            "perception": "none",
            "planner": cfg["planner"],
            "localization": cfg["localization"],
            "map_file": cfg["map_file"],
            "lidar_safety": "true",
            "cloud_in_topic": cfg["cloud_in_topic"],
            "publish_lidar_extrinsic": cfg["publish_lidar_extrinsic"],
            "hardware_adapter": cfg["hardware_adapter"],
            "dry_run_log_path": cfg["dry_run_log_path"],
            "use_sim_time": cfg["use_sim_time"],
            "log_level": cfg["log_level"],
        }.items(),
    )

    semantic = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("go2_semantic_bringup"), "launch", "semantic_nav.launch.py"])]
        ),
        launch_arguments={
            "enable_detector": cfg["enable_detector"],
            "enable_scene_graph": cfg["enable_scene_graph"],
            "enable_grounding": "true",
            "allow_goal_publication": cfg["allow_goal_publication"],
            # The staged planner runs no Nav2, so there is no costmap; grounding
            # then samples stand-offs without one (the LiDAR hazard source and
            # the arbiter still guard motion; the staged planner cannot avoid
            # obstacles, so use it only for straight-line-clear goals).
            "use_costmap_gate": PythonExpression(
                ["'false' if '", cfg["planner"], "' == 'staged_nav' else 'true'"]
            ),
            "navigation_backend": "nav2_action",
            "navigate_to_pose_action": "/navigate_to_pose",
            "image_topic": _per_mode(cfg, "image_topic", 0),
            "depth_topic": _per_mode(cfg, "depth_topic", 1),
            "camera_info_topic": _per_mode(cfg, "camera_info_topic", 2),
            "detector_params": detector_params,
            "use_slam_fallback": "false",
            "use_sim_time": cfg["use_sim_time"],
            "log_level": cfg["log_level"],
        }.items(),
    )

    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default) for name, default in _ARGS.items()]
        + calibration_args
        + [base, rgb_lidar, semantic]
    )
