# - Try to find fmtlib
# Once done this will define
#  FMT_FOUND - System has fmtlib
#  FMT_INCLUDE_DIRS - The fmtlib include directories
#  FMT_LIBRARIES - The libraries needed to use fmtlib
#  FMT_DEFINITIONS - Compiler switches required for using fmtlib

# Modified https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/How-To-Find-Libraries

find_package(PkgConfig)
pkg_check_modules(PKGCONF QUIET fmt)
set(FMT_DEFINITIONS ${PKGCONF_FMT_CFLAGS_OTHER})

find_path(FMT_INCLUDE_DIR fmt/format.h
        HINTS ${PKGCONF_FMT_INCLUDEDIR} ${PKGCONF_FMT_INCLUDE_DIRS}
        PATH_SUFFIXES fmt)

find_library(FMT_LIBRARY NAMES fmt
        HINTS ${PKGCONF_FMT_LIBDIR} ${PKGCONF_FMT_LIBRARY_DIRS} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(fmt DEFAULT_MSG
        FMT_LIBRARY FMT_INCLUDE_DIR)

mark_as_advanced(FMT_INCLUDE_DIR FMT_LIBRARY )

set(FMT_LIBRARIES  ${FMT_LIBRARY} )
set(FMT_INCLUDE_DIRS ${FMT_INCLUDE_DIR} )