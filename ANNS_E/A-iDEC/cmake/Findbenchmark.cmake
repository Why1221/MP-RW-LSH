# - Try to find Google benchmark
# Once done this will define
#  BENCHMARK_FOUND - System has google benchmark
#  BENCHMARK_INCLUDE_DIRS - The google benchmark include directories
#  BENCHMARK_LIBRARIES - The libraries needed to use google benchmark
#  BENCHMARK_DEFINITIONS - Compiler switches required for using google benchmark

# Modified https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/How-To-Find-Libraries

find_package(PkgConfig)
pkg_check_modules(Google QUIET benchmark)
set(BENCHMARK_DEFINITIONS ${Google_BENCHMARK_CFLAGS_OTHER})

find_path(BENCHMARK_INCLUDE_DIR benchmark/benchmark.h
        HINTS ${Google_BENCHMARK_INCLUDEDIR} ${Google_BENCHMARK_INCLUDE_DIRS}
        PATH_SUFFIXES benchmark)

find_library(BENCHMARK_LIBRARY NAMES benchmark
        HINTS ${Google_BENCHMARK_LIBDIR} ${Google_BENCHMARK_LIBRARY_DIRS} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(benchmark  DEFAULT_MSG
        BENCHMARK_LIBRARY BENCHMARK_INCLUDE_DIR)

mark_as_advanced(BENCHMARK_INCLUDE_DIR BENCHMARK_LIBRARY )

set(BENCHMARK_LIBRARIES  ${BENCHMARK_LIBRARY} )
set(BENCHMARK_INCLUDE_DIRS ${BENCHMARK_INCLUDE_DIR} )
