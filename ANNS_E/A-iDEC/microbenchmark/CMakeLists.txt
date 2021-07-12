
MACRO(ADD_GOOGLE_BENCHMARK _NAME _EXTRA_LIBS)
    if (DEBUG_ENABLED)
        message("add test with name ${_NAME} depending on: ${_EXTRA_LIBS}")
    endif()
    ADD_EXECUTABLE(${_NAME} ${_NAME}.cpp)
    if (WIN32)
        # TODO fix lib linking issues on WINDOWS
        TARGET_LINK_LIBRARIES(${_NAME} PRIVATE ${_EXTRA_LIBS} ${BENCHMARK_LIBRARIES})
    else ()
        TARGET_LINK_LIBRARIES(${_NAME} ${_EXTRA_LIBS} ${BENCHMARK_LIBRARIES})
    endif ()
ENDMACRO()

ADD_GOOGLE_BENCHMARK(bench_cgk_impl "")