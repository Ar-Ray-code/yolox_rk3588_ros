cmake_minimum_required(VERSION 3.10)
project(yolox_rk3588_ros)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED)

ament_auto_add_library(yolox_rk3588_ros SHARED
    src/yolox_rk3588_ros.cpp
)
ament_target_dependencies(yolox_rk3588_ros OpenCV)

rclcpp_components_register_node(yolox_rk3588_ros
    PLUGIN "yolox_rk3588_ros::YOLOX_ROS"
    EXECUTABLE yolox_rk3588_ros_node
)


include_directories(include)
ament_auto_package()
