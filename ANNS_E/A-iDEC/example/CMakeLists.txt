# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.2)


# ADD_MY_TEST macro
MACRO(ADD_EXAMPLE _NAME _EXTRA_LIBS)
    message("add an example with name ${_NAME} depending on: ${_EXTRA_LIBS}")
    ADD_EXECUTABLE(${_NAME} ${_NAME}.cpp)
    if (WIN32)
        # TODO fix windows lib linking issues
        TARGET_LINK_LIBRARIES(${_NAME} PRIVATE ${_EXTRA_LIBS})
    else ()
        TARGET_LINK_LIBRARIES(${_NAME} ${_EXTRA_LIBS})
    endif ()
ENDMACRO()


# add examples
ADD_EXAMPLE(is_same_example "")
