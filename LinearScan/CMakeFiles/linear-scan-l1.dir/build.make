# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan

# Include any dependencies generated for this target.
include CMakeFiles/linear-scan-l1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/linear-scan-l1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/linear-scan-l1.dir/flags.make

CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o: CMakeFiles/linear-scan-l1.dir/flags.make
CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o: linear_scan_main_l1.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o -c /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/linear_scan_main_l1.cc

CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/linear_scan_main_l1.cc > CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.i

CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/linear_scan_main_l1.cc -o CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.s

CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o.requires:

.PHONY : CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o.requires

CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o.provides: CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o.requires
	$(MAKE) -f CMakeFiles/linear-scan-l1.dir/build.make CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o.provides.build
.PHONY : CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o.provides

CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o.provides.build: CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o


CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o: CMakeFiles/linear-scan-l1.dir/flags.make
CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o: linear_scan_l1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o -c /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/linear_scan_l1.cpp

CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/linear_scan_l1.cpp > CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.i

CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/linear_scan_l1.cpp -o CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.s

CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o.requires:

.PHONY : CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o.requires

CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o.provides: CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o.requires
	$(MAKE) -f CMakeFiles/linear-scan-l1.dir/build.make CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o.provides.build
.PHONY : CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o.provides

CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o.provides.build: CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o


# Object files for target linear-scan-l1
linear__scan__l1_OBJECTS = \
"CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o" \
"CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o"

# External object files for target linear-scan-l1
linear__scan__l1_EXTERNAL_OBJECTS =

linear-scan-l1: CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o
linear-scan-l1: CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o
linear-scan-l1: CMakeFiles/linear-scan-l1.dir/build.make
linear-scan-l1: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
linear-scan-l1: /usr/lib/x86_64-linux-gnu/libpthread.so
linear-scan-l1: /usr/lib/x86_64-linux-gnu/libsz.so
linear-scan-l1: /usr/lib/x86_64-linux-gnu/libz.so
linear-scan-l1: /usr/lib/x86_64-linux-gnu/libdl.so
linear-scan-l1: /usr/lib/x86_64-linux-gnu/libm.so
linear-scan-l1: CMakeFiles/linear-scan-l1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable linear-scan-l1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/linear-scan-l1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/linear-scan-l1.dir/build: linear-scan-l1

.PHONY : CMakeFiles/linear-scan-l1.dir/build

CMakeFiles/linear-scan-l1.dir/requires: CMakeFiles/linear-scan-l1.dir/linear_scan_main_l1.cc.o.requires
CMakeFiles/linear-scan-l1.dir/requires: CMakeFiles/linear-scan-l1.dir/linear_scan_l1.cpp.o.requires

.PHONY : CMakeFiles/linear-scan-l1.dir/requires

CMakeFiles/linear-scan-l1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/linear-scan-l1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/linear-scan-l1.dir/clean

CMakeFiles/linear-scan-l1.dir/depend:
	cd /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan /media/gtnetuser/Dell/ann-codes-multi-probe/ann-codes-rw/ann-codes/external-memory/LinearScan/CMakeFiles/linear-scan-l1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/linear-scan-l1.dir/depend

