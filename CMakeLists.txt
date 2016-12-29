cmake_minimum_required(VERSION 2.8.11)
project(DISPARITY_MAP)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

set(LIBELAS_SRC_DIR elas)
set(DAISY_DESC_INC_DIR ./daisy_descriptor/include)
SET(CMAKE_CXX_FLAGS "-msse3")

# include directory
include_directories("${LIBELAS_SRC_DIR}")
include_directories("${DAISY_DESC_INC_DIR}")

# sources
FILE(GLOB LIBELAS_SRC_FILES "elas/*.cpp")
FILE(GLOB DAISY_DESC_SRC_FILES "daisy_descriptor/src/*")

add_executable(disparity-daisy ${DAISY_DESC_SRC_FILES} disparity-daisy.cpp)
target_link_libraries(disparity-daisy ${OpenCV_LIBS})

add_executable(disparity-elas ${LIBELAS_SRC_FILES} disparity-elas.cpp)
target_link_libraries(disparity-elas ${OpenCV_LIBS})

add_executable(epipolar epipolar.cpp popt_pp.h)
target_link_libraries(epipolar ${OpenCV_LIBS} "-lpopt")

add_executable(disparity disparity.cpp popt_pp.h)
target_link_libraries(disparity ${OpenCV_LIBS} "-lpopt")

add_executable(disparity-lbp disparity-lbp.cpp popt_pp.h)
target_link_libraries(disparity-lbp ${OpenCV_LIBS} "-lpopt")

add_executable(disparity-orb disparity-orb.cpp popt_pp.h)
target_link_libraries(disparity-orb ${OpenCV_LIBS} "-lpopt")

add_executable(reconstruct reconstruct.cpp popt_pp.h)
target_link_libraries(reconstruct ${OpenCV_LIBS} "-lpopt")