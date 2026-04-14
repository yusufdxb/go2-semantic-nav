cmake_minimum_required(VERSION 3.8)
project(go2_semantic_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)

set(msg_files
  "msg/SemanticDetection.msg"
  "msg/SemanticDetectionArray.msg"
  "msg/SemanticObject.msg"
  "msg/SceneGraphEdge.msg"
  "msg/SceneGraph.msg"
  "msg/GroundingCandidate.msg"
)

set(srv_files
  "srv/QueryObjects.srv"
)

set(action_files
  "action/GroundAndNavigate.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    vision_msgs
    builtin_interfaces
    action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
