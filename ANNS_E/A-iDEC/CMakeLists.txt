# CMakeList.txt : Top-level CMake project file, do global configuration
#
cmake_minimum_required(VERSION 3.2)

# set project name
project("StringSimilarity")
# set the language standard
set(CMAKE_CXX_STANDARD 17)

# set project root path
set(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# Options
option(USE_BOOST "enable Boost Support" ON)
option(BUILD_TEST "build tests" ON)
option(BUILD_BENCHMARK "build benchmarks" ON)
option(USE_GOOGLE_BENCHMARK "enable Google Benchmark" ON)
option(USE_FMTLIB "enable fmtlib" ON)


# Enable test
if (BUILD_TEST)
    message("Tests are enabled")
    enable_testing()
endif ()

# check build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOUPPER ${CMAKE_BUILD_TYPE} SS_Build_Type)
if (${SS_Build_Type} MATCHES "DEBUG")
    message("Enable debug by adding DEBUG macro")
    set(DEBUG_ENABLED ON)
    add_definitions(-DDEBUG)
endif()

# find boost 
if (USE_BOOST)
    find_package(Boost QUIET REQUIRED COMPONENTS filesystem)
    add_definitions(-DUSE_BOOST)
    if (DEBUG_ENABLED)
        message("boost include dir: ${Boost_INCLUDE_DIR}")
    endif()
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
endif ()

if (USE_FMTLIB)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(format QUIET REQUIRED)
    if (DEBUG_ENABLED)
        message("fmt include dir: ${FMT_INCLUDE_DIRS}, lib: ${FMT_LIBRARIES}")
     endif()
    add_definitions(-DUSE_FMTLIB)
endif()
#    if (WIN32)
#        find_package(fmt CONFIG REQUIRED)
#        list(APPEND SS_PRIVATE_LIBRARIES "fmt::fmt fmt::fmt-header-only")
#    else ()
#        list(APPEND SS_EXTRA_LIBRARIES fmt)
#    endif ()
#endif()

# Add extra include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party)


# Enable benchmark (part of benchmarks using Google benchmark)
if (BUILD_BENCHMARK)
    if (USE_GOOGLE_BENCHMARK)
        # fix issues on windows
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
		find_package(benchmark REQUIRED)
		# list(APPEND ${SS_EXTRA_LIBRARIES} ${BENCHMARK_LIBRARIES})
        if (UNIX)
            list(APPEND BENCHMARK_LIBRARIES pthread)
        endif()
        if (DEBUG_ENABLED)
            message("Google benchmark: include dir: ${BENCHMARK_INCLUDE_DIR}, libraries: ${BENCHMARK_LIBRARIES}")
        endif ()
		include_directories(${BENCHMARK_INCLUDE_DIR})
        add_definitions(-DUSE_GOOLE_BENCHMARK)
    endif()
else()
    if (USE_GOOGLE_BENCHMARK)
        message(WARNING "Google benchmark will not be used, as BUILD_BENCHMARK option is disabled")
    endif()
endif()


# build 3rd party library
# BUILD_3RD_LIBRARY macro
MACRO(BUILD_3RD_LIBRARY _NAME _SRC)
    if (DEBUG_ENABLED)
    message("Build lib ${_NAME} with sources: ${_SRC}")
    endif()
    ADD_LIBRARY(${_NAME} STATIC ${_SRC})
ENDMACRO()

# SRS related
set(ProjDataSrc 3rd_party/srs/ProjData.h 3rd_party/srs/ProjData.cpp)
BUILD_3RD_LIBRARY(proj_data "${ProjDataSrc}")
set(CoverTreeSrc 3rd_party/srs/SRSCoverTree.h 3rd_party/srs/SRSCoverTree.cpp)
BUILD_3RD_LIBRARY(cover_tree "${CoverTreeSrc}")
set(SRS_LIBRARIES "proj_data" "cover_tree")
# edlib
set(EdlibSrc 3rd_party/edlib/edlib.h 3rd_party/edlib/edlib.cpp)
BUILD_3RD_LIBRARY(edlib "${EdlibSrc}")
add_definitions(-DUSE_EDLIB)

message("Extra libaries: ${SS_EXTRA_LIBRARIES}")

##
if (WIN32)
    find_package(hdf5 CONFIG REQUIRED)
    set(HDF5_C_LIBRARIES "hdf5::hdf5-shared" "hdf5::hdf5-static" "hdf5::hdf5_hl-shared" "hdf5::hdf5_hl-static")
else ()
    find_package(HDF5 QUIET REQUIRED)
    if (DEBUG_ENABLED)
        message("HDF5 include : ${HDF5_INCLUDE_DIRS} libraries: ${HDF5_C_LIBRARIES}")
    endif()
    include_directories(${HDF5_INCLUDE_DIRS})
endif ()


link_directories(${CMAKE_CURRENT_BINARY_DIR})
# Include sub-projects.
if (BUILD_TEST)
    add_subdirectory(test)
endif ()
add_subdirectory(example)
if (BUILD_BENCHMARK)
    add_subdirectory(benchmark)
    if(USE_GOOGLE_BENCHMARK)
        add_subdirectory(microbenchmark)
    endif()
endif()

add_subdirectory(temporary)
