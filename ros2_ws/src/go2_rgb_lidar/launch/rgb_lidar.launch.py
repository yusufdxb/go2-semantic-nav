"""GO2 front camera + LiDAR-projected aligned depth.

    ros2 launch go2_rgb_lidar rgb_lidar.launch.py multicast_iface:=enP8p1s0

Publishes /camera/front/image_raw, /camera/front/lidar_depth and
/camera/front/camera_info for the open-vocab detector. Needs the base stack's
LiDAR relay (/go2/lidar/points) and odom->base_link TF.

Arguments
---------
intrinsics_file            camera_calibration YAML (default: shipped NOMINAL file)
extrinsics_file            camera optical frame pose in base_link (default: NOMINAL)
allow_nominal_calibration  false: depth is withheld until both files are measured
publish_overlay            true to publish /camera/front/lidar_overlay for calibration
enable_camera              false when another node already publishes the image
camera_impl                cpp (default: go2_front_camera_cpp, stamps at packet
                           arrival, ~2 ms lower latency and half the CPU of the
                           Python driver on the desktop benchmark) | py
camera_decoder             cpp only: nvv4l2 (Jetson hardware, default) | avdec
multicast_iface            robot-side NIC on the Jetson (empty = OS default)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("go2_rgb_lidar")
    cfg = os.path.join(share, "config", "rgb_lidar.yaml")
    args = [
        DeclareLaunchArgument("intrinsics_file",
                              default_value=os.path.join(share, "config", "front_camera_intrinsics_nominal.yaml")),
        DeclareLaunchArgument("extrinsics_file",
                              default_value=os.path.join(share, "config", "front_camera_extrinsics_nominal.yaml")),
        DeclareLaunchArgument("allow_nominal_calibration", default_value="false"),
        DeclareLaunchArgument("publish_overlay", default_value="false"),
        DeclareLaunchArgument("enable_camera", default_value="true"),
        DeclareLaunchArgument("camera_impl", default_value="cpp"),
        DeclareLaunchArgument("camera_decoder", default_value="nvv4l2"),
        DeclareLaunchArgument("multicast_iface", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    def camera_enabled(impl: str):
        return IfCondition(PythonExpression(
            ["'", LaunchConfiguration("enable_camera"), "' == 'true' and '", LaunchConfiguration("camera_impl"),
             f"' == '{impl}'"]))

    camera_py = Node(
        package="go2_rgb_lidar",
        executable="front_camera_node",
        name="go2_front_camera",
        output="screen",
        parameters=[cfg, {"multicast_iface": LaunchConfiguration("multicast_iface"),
                          "use_sim_time": use_sim_time}],
        condition=camera_enabled("py"),
    )
    cpp_cfg = os.path.join(get_package_share_directory("go2_front_camera_cpp"), "config", "front_camera.yaml")
    camera_cpp = Node(
        package="go2_front_camera_cpp",
        executable="front_camera_node",
        name="go2_front_camera",
        output="screen",
        parameters=[cpp_cfg, {"multicast_iface": LaunchConfiguration("multicast_iface"),
                              "decoder": LaunchConfiguration("camera_decoder"),
                              "use_sim_time": use_sim_time}],
        condition=camera_enabled("cpp"),
    )
    depth = Node(
        package="go2_rgb_lidar",
        executable="lidar_depth_node",
        name="go2_lidar_depth",
        output="screen",
        parameters=[
            cfg,
            {
                "intrinsics_file": LaunchConfiguration("intrinsics_file"),
                "extrinsics_file": LaunchConfiguration("extrinsics_file"),
                "allow_nominal_calibration": ParameterValue(
                    LaunchConfiguration("allow_nominal_calibration"), value_type=bool),
                "publish_overlay": ParameterValue(LaunchConfiguration("publish_overlay"), value_type=bool),
                "use_sim_time": use_sim_time,
            },
        ],
    )
    return LaunchDescription(args + [camera_py, camera_cpp, depth])
