cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

project(linear-scan-l1)

include_directories(../Common)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/HighFive)
add_executable(linear-scan-l1 linear_scan_main_l1.cc linear_scan_l1.cpp)
target_link_libraries(linear-scan-l1 HighFive)
