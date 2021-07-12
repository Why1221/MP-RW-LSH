cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(edlib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(lib_src ${PROJECT_ROOT_PATH}/third-party/edlib/edlib.cpp)

# Create libraries.

## shared
add_library(edlib SHARED ${lib_src})

target_include_directories(edlib PUBLIC
        $<BUILD_INTERFACE:${PROJECT_ROOT_PATH}/third-party>
        $<INSTALL_INTERFACE:include>)

## static
add_library(edlib_static STATIC ${lib_src})
target_include_directories(edlib_static PUBLIC
        $<BUILD_INTERFACE:${PROJECT_ROOT_PATH}/third-party>
        $<INSTALL_INTERFACE:include>)

#add_executable(helloWorld
#        helloWorld.cpp)
#target_link_libraries(helloWorld edlib_static)
# Create target 'install' for installing libraries.
install(TARGETS edlib edlib_static DESTINATION lib)