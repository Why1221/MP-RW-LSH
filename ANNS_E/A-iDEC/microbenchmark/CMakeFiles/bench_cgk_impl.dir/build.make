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
include microbenchmark/CMakeFiles/bench_cgk_impl.dir/depend.make

# Include the progress variables for this target.
include microbenchmark/CMakeFiles/bench_cgk_impl.dir/progress.make

# Include the compile flags for this target's objects.
include microbenchmark/CMakeFiles/bench_cgk_impl.dir/flags.make

microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o: microbenchmark/CMakeFiles/bench_cgk_impl.dir/flags.make
microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o: microbenchmark/bench_cgk_impl.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/gtnetuser/Dell/StringSimilarity/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o"
	cd /media/gtnetuser/Dell/StringSimilarity/microbenchmark && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o -c /media/gtnetuser/Dell/StringSimilarity/microbenchmark/bench_cgk_impl.cpp

microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.i"
	cd /media/gtnetuser/Dell/StringSimilarity/microbenchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/gtnetuser/Dell/StringSimilarity/microbenchmark/bench_cgk_impl.cpp > CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.i

microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.s"
	cd /media/gtnetuser/Dell/StringSimilarity/microbenchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/gtnetuser/Dell/StringSimilarity/microbenchmark/bench_cgk_impl.cpp -o CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.s

microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o.requires:

.PHONY : microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o.requires

microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o.provides: microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o.requires
	$(MAKE) -f microbenchmark/CMakeFiles/bench_cgk_impl.dir/build.make microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o.provides.build
.PHONY : microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o.provides

microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o.provides.build: microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o


# Object files for target bench_cgk_impl
bench_cgk_impl_OBJECTS = \
"CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o"

# External object files for target bench_cgk_impl
bench_cgk_impl_EXTERNAL_OBJECTS =

microbenchmark/bench_cgk_impl: microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o
microbenchmark/bench_cgk_impl: microbenchmark/CMakeFiles/bench_cgk_impl.dir/build.make
microbenchmark/bench_cgk_impl: /usr/local/lib/libbenchmark.a
microbenchmark/bench_cgk_impl: microbenchmark/CMakeFiles/bench_cgk_impl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/gtnetuser/Dell/StringSimilarity/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bench_cgk_impl"
	cd /media/gtnetuser/Dell/StringSimilarity/microbenchmark && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_cgk_impl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
microbenchmark/CMakeFiles/bench_cgk_impl.dir/build: microbenchmark/bench_cgk_impl

.PHONY : microbenchmark/CMakeFiles/bench_cgk_impl.dir/build

microbenchmark/CMakeFiles/bench_cgk_impl.dir/requires: microbenchmark/CMakeFiles/bench_cgk_impl.dir/bench_cgk_impl.cpp.o.requires

.PHONY : microbenchmark/CMakeFiles/bench_cgk_impl.dir/requires

microbenchmark/CMakeFiles/bench_cgk_impl.dir/clean:
	cd /media/gtnetuser/Dell/StringSimilarity/microbenchmark && $(CMAKE_COMMAND) -P CMakeFiles/bench_cgk_impl.dir/cmake_clean.cmake
.PHONY : microbenchmark/CMakeFiles/bench_cgk_impl.dir/clean

microbenchmark/CMakeFiles/bench_cgk_impl.dir/depend:
	cd /media/gtnetuser/Dell/StringSimilarity && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/gtnetuser/Dell/StringSimilarity /media/gtnetuser/Dell/StringSimilarity/microbenchmark /media/gtnetuser/Dell/StringSimilarity /media/gtnetuser/Dell/StringSimilarity/microbenchmark /media/gtnetuser/Dell/StringSimilarity/microbenchmark/CMakeFiles/bench_cgk_impl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : microbenchmark/CMakeFiles/bench_cgk_impl.dir/depend

