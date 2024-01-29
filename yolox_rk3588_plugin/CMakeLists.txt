cmake_minimum_required(VERSION 3.8)
project(yolox_rk3588_plugin)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(OpenCV REQUIRED)

file(GLOB_RECURSE SRCS src/*.cpp)
ament_auto_add_library(yolox_rk3588_plugin SHARED
  ${SRCS}
)
ament_target_dependencies(${PROJECT_NAME} OpenCV)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

pluginlib_export_plugin_description_file(detector2d_base yolox_rk3588_plugin.xml)

include_directories(include)
ament_auto_package()