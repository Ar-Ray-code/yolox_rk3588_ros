cmake_minimum_required(VERSION 3.10)
project(yolox_rk3588_cpp)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_STANDARD 17)
set(TARGET_SOC rk3588)
set(CMAKE_SYSTEM_NAME Linux)
set(TARGET_LIB_ARCH aarch64)
set(TARGET_SOC rk3588)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB LIBS_SO ${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/*.so)

ament_auto_add_executable(${PROJECT_NAME}
    src/main.cpp
    src/postprocess.cpp
    src/yolox.cpp
    src/image_utils.cpp
    src/file_utils.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${LIBS_SO}
    Threads::Threads
)
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/
    DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

include_directories(include)

ament_auto_package()
