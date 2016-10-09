cmake_minimum_required(VERSION 2.8.11)
project(DISPARITY_MAP)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(descriptor descriptor.cpp popt_pp.h)
target_link_libraries(descriptor ${OpenCV_LIBS} "-lpopt")

add_executable(disparity disparity.cpp popt_pp.h)
target_link_libraries(disparity ${OpenCV_LIBS} "-lpopt")