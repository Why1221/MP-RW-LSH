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
CMAKE_SOURCE_DIR = /media/gtnetuser/Dell/StringSimilarity

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/gtnetuser/Dell/StringSimilarity

# Include any dependencies generated for this target.
include benchmark/CMakeFiles/sp_bench_1nn_all.dir/depend.make

# Include the progress variables for this target.
include benchmark/CMakeFiles/sp_bench_1nn_all.dir/progress.make

# Include the compile flags for this target's objects.
include benchmark/CMakeFiles/sp_bench_1nn_all.dir/flags.make

benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o: benchmark/CMakeFiles/sp_bench_1nn_all.dir/flags.make
benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o: benchmark/sp_bench_1nn_all.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/gtnetuser/Dell/StringSimilarity/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o"
	cd /media/gtnetuser/Dell/StringSimilarity/benchmark && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o -c /media/gtnetuser/Dell/StringSimilarity/benchmark/sp_bench_1nn_all.cpp

benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.i"
	cd /media/gtnetuser/Dell/StringSimilarity/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/gtnetuser/Dell/StringSimilarity/benchmark/sp_bench_1nn_all.cpp > CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.i

benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.s"
	cd /media/gtnetuser/Dell/StringSimilarity/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/gtnetuser/Dell/StringSimilarity/benchmark/sp_bench_1nn_all.cpp -o CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.s

benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o.requires:

.PHONY : benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o.requires

benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o.provides: benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o.requires
	$(MAKE) -f benchmark/CMakeFiles/sp_bench_1nn_all.dir/build.make benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o.provides.build
.PHONY : benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o.provides

benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o.provides.build: benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o


# Object files for target sp_bench_1nn_all
sp_bench_1nn_all_OBJECTS = \
"CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o"

# External object files for target sp_bench_1nn_all
sp_bench_1nn_all_EXTERNAL_OBJECTS =

benchmark/sp_bench_1nn_all: benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o
benchmark/sp_bench_1nn_all: benchmark/CMakeFiles/sp_bench_1nn_all.dir/build.make
benchmark/sp_bench_1nn_all: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
benchmark/sp_bench_1nn_all: /usr/lib/x86_64-linux-gnu/libboost_system.so
benchmark/sp_bench_1nn_all: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
benchmark/sp_bench_1nn_all: /usr/lib/x86_64-linux-gnu/libpthread.so
benchmark/sp_bench_1nn_all: /usr/lib/x86_64-linux-gnu/libsz.so
benchmark/sp_bench_1nn_all: /usr/lib/x86_64-linux-gnu/libz.so
benchmark/sp_bench_1nn_all: /usr/lib/x86_64-linux-gnu/libdl.so
benchmark/sp_bench_1nn_all: /usr/lib/x86_64-linux-gnu/libm.so
benchmark/sp_bench_1nn_all: libproj_data.a
benchmark/sp_bench_1nn_all: libcover_tree.a
benchmark/sp_bench_1nn_all: /usr/local/lib/libfmt.a
benchmark/sp_bench_1nn_all: benchmark/CMakeFiles/sp_bench_1nn_all.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/gtnetuser/Dell/StringSimilarity/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sp_bench_1nn_all"
	cd /media/gtnetuser/Dell/StringSimilarity/benchmark && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sp_bench_1nn_all.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmark/CMakeFiles/sp_bench_1nn_all.dir/build: benchmark/sp_bench_1nn_all

.PHONY : benchmark/CMakeFiles/sp_bench_1nn_all.dir/build

benchmark/CMakeFiles/sp_bench_1nn_all.dir/requires: benchmark/CMakeFiles/sp_bench_1nn_all.dir/sp_bench_1nn_all.cpp.o.requires

.PHONY : benchmark/CMakeFiles/sp_bench_1nn_all.dir/requires

benchmark/CMakeFiles/sp_bench_1nn_all.dir/clean:
	cd /media/gtnetuser/Dell/StringSimilarity/benchmark && $(CMAKE_COMMAND) -P CMakeFiles/sp_bench_1nn_all.dir/cmake_clean.cmake
.PHONY : benchmark/CMakeFiles/sp_bench_1nn_all.dir/clean

benchmark/CMakeFiles/sp_bench_1nn_all.dir/depend:
	cd /media/gtnetuser/Dell/StringSimilarity && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/gtnetuser/Dell/StringSimilarity /media/gtnetuser/Dell/StringSimilarity/benchmark /media/gtnetuser/Dell/StringSimilarity /media/gtnetuser/Dell/StringSimilarity/benchmark /media/gtnetuser/Dell/StringSimilarity/benchmark/CMakeFiles/sp_bench_1nn_all.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmark/CMakeFiles/sp_bench_1nn_all.dir/depend

