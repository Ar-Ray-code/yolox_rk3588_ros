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

ament_auto_add_library(${PROJECT_NAME} SHARED
    src/postprocess.cpp
    src/yolox.cpp
    src/image_utils.cpp
    src/file_utils.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    /usr/local/lib/librga.so
    /usr/local/lib/librknnrt.so
    Threads::Threads
)

add_executable(${PROJECT_NAME}_exec
    src/main.cpp
)
target_link_libraries(${PROJECT_NAME}_exec PRIVATE
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    /usr/local/lib/librga.so
    /usr/local/lib/librknnrt.so
    Threads::Threads
)

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_exec
    DESTINATION lib/${PROJECT_NAME}
)

include_directories(include)
ament_auto_package()
