# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.2)

# DATE & TIME macro (Copied from TODO)
# Return the date (yyyy-mm-dd)
macro(DATE RESULT)
    if(WIN32)
        execute_process(COMMAND "cmd" " /C date /T" OUTPUT_VARIABLE ${RESULT})
        string(REGEX REPLACE "(..)/(..)/(....).*" "\\3-\\2-\\1" ${RESULT} ${${RESULT}})
    elseif(UNIX)
        execute_process(COMMAND "date" "+%Y-%m-%d" OUTPUT_VARIABLE ${RESULT})
        # Strip lead and trailing whitepasce
        string(STRIP "${${RESULT}}" ${RESULT})
        message("current date: ${${RESULT}}")
    else()
        message(SEND_ERROR "Unable to detect date")
        set(${RESULT} UNKNOWN)
    endif()
endmacro()

# Return the time (hh:mm:ss)
macro(TIME RESULT)
    if(WIN32)
        execute_process(COMMAND "cmd" " /C echo %TIME%" OUTPUT_VARIABLE ${RESULT})
        string(REGEX REPLACE "(..:..:..),(..)" "\\1" ${RESULT} ${${RESULT}})
    elseif(UNIX)
        execute_process(COMMAND "date" "+%H:%M:%S" OUTPUT_VARIABLE ${RESULT})
    else()
        message(SEND_ERROR "Unable to detect time")
        set(${RESULT} UNKNOWN)
    endif()
endmacro()

#################################################################################################
## benchmark input & output configurations
set(DATA_PATH "${PROJECT_ROOT_PATH}/data/Hamming/compact/")
set(DATA_SETS "")
# million-level data sets
list(APPEND DATA_SETS "word2bits-800-hamming.hdf5") #WORD2BITS 0.4M Hamming
list(APPEND DATA_SETS "sift-256-hamming.hdf5") #SIFT1M Hamming

# billion-level data sets
DATE(CURRENT_DATE)
set(RESULT_PATH "${PROJECT_ROOT_PATH}/results/${CURRENT_DATE}/million-level/")
set(INDEX_PATH_REL "temporary")
set(RAW_DATASET "base_compact")
set(QUERY_DATASET "query_compact")
set(GND_DIST_DATASET "ground_truth_dist")
set(WORDSIZE_DATASET "word_size")
set(KNN_K "K")

# billion-level data sets
set(DATA_PATH_BL "/Data/Dropbox/HUGE-DATA/")
set(DATA_SETS_BL "")
# million-level data sets
list(APPEND DATA_SETS_BL "Hamming_mih_128_sift_1B.hdf5") #SIFT1B Hamming
list(APPEND DATA_SETS_BL "Hamming_mih_128_gist_80M.hdf5") #GIST80M Hamming
#list(APPEND DATA_SETS_BL "Hamming_mih_256_gist_80M.hdf5") #GIST80M Hamming
#list(APPEND DATA_SETS_BL "Hamming_mih_256_sift_1B.hdf5") #SIFT1B Hamming

# billion-level data sets
set(RESULT_PATH_BL "${PROJECT_ROOT_PATH}/results/${CURRENT_DATE}/billion-level/")
set(INDEX_PATH_REL_BL "temporary")
set(RAW_DATASET_PREFIX_BL "base/BLK_")
set(QUERY_DATASET_BL "query")
set(RAW_DATASET_BLOCKSIZE_BL "size_block")
set(RAW_DATASET_NBLOCKS_BL "number_blocks")

##
configure_file(${PROJECT_ROOT_PATH}/config/benchmark_config.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/benchmark_config.h
        @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
#######################################################################################################


# ADD_MY_TEST macro
MACRO(ADD_BENCHMARK _NAME _EXTRA_LIBS)
    if (DEBUG_ENABLED)
    message("add test with name ${_NAME} depending on: ${_EXTRA_LIBS}")
    endif()
    ADD_EXECUTABLE(${_NAME} ${_NAME}.cpp)
    if (WIN32)
        # TODO fix lib linking issues on WINDOWS
        TARGET_LINK_LIBRARIES(${_NAME} PRIVATE ${_EXTRA_LIBS})
    else ()
        TARGET_LINK_LIBRARIES(${_NAME} ${_EXTRA_LIBS})
    endif ()
ENDMACRO()

set(BENCHMARK_LIBRARIES "${Boost_LIBRARIES}" "${HDF5_C_LIBRARIES}" "${SRS_LIBRARIES}" "${FMT_LIBRARIES}")

ADD_BENCHMARK(sp_bench_brute_force "${BENCHMARK_LIBRARIES}")
ADD_BENCHMARK(old_benchmark_brute_force "${BENCHMARK_LIBRARIES}")
ADD_BENCHMARK(sp_bench_1nn_all "${BENCHMARK_LIBRARIES}")

