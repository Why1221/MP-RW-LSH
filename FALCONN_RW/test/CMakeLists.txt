cmake_minimum_required(VERSION 3.0)

project(test)

#include_directories(../Common)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
#add_library(lib)
#target_include_directories(lib INTERFACE external/eigen)
#target_include_directories(lib INTERFACE external/simple-serializer)
#target_include_directories(lib INTERFACE external/nlohmann)
#target_include_directories(lib INTERFACE external/wyhash)

include_directories(../src/include/falconn/core)
#include_directories(../external/eigen)
#include_directories(../external/simple-serializer)
#include_directories(../external/nlohmann)
include_directories(../external)
include_directories(../feigenbaum)

#add_executable(${PROJECT_NAME} test.cc)
add_executable(${PROJECT_NAME} tf.cc)
# target_link_libraries(${PROJECT_NAME} lib)
#target_link_libraries(${PROJECT_NAME} pthread)

