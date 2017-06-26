cmake_minimum_required(VERSION 3.6)
project(UniversalBlender C CXX)

if (APPLE)
message(STATUS "Configuring UniversalBlender project for macOS...")
elseif (WIN32)
message(STATUS "Configuring UniversalBlender project for Windows...")
endif(APPLE)
################################################################
# Global constants
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(VERSION_INFO "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

################################################################
# Output project information
message(STATUS "UniversalBlender source path: ${PROJECT_SOURCE_DIR}")
message(STATUS "UniversalBlender binary path: ${PROJECT_BINARY_DIR}")

################################################################
# option
option(UniveralBlender_BUILD_TESTS "Build test project" ON)
# sub projects
add_subdirectory(src)
add_subdirectory(wrapper)
add_subdirectory(test)
