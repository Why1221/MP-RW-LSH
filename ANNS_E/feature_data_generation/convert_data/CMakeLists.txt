# set the language standard
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(foo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_INCLUDE_CURRENT_DIR ON)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/HighFive)
add_executable(convert convert.cpp edlib.cpp)
target_link_libraries(convert HighFive)
include_directories("edlib")
find_package(Boost 1.58.0 REQUIRED COMPONENTS system filesystem)
target_link_libraries(convert ${Boost_LIBRARIES})
