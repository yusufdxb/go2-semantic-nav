cmake_minimum_required(VERSION 3.16)
project(go2_front_camera_cpp LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

add_library(front_camera_pipeline src/pipeline.cpp)
target_include_directories(front_camera_pipeline PUBLIC include)

add_executable(front_camera_node src/front_camera_node.cpp)
target_link_libraries(front_camera_node front_camera_pipeline PkgConfig::GST)
ament_target_dependencies(front_camera_node rclcpp sensor_msgs std_msgs)

install(TARGETS front_camera_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_pipeline test/test_pipeline.cpp)
  target_link_libraries(test_pipeline front_camera_pipeline PkgConfig::GST)
endif()

ament_package()
