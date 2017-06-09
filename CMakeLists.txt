cmake_minimum_required(VERSION 3.6)
project(UniversalBlender)

################################################################
# Global constants
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(VERSION_INFO "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

################################################################
# Output project information
message("UniversalBlender source path: ${PROJECT_SOURCE_DIR}")
message("UniversalBlender binary path: ${PROJECT_BINARY_DIR}")

################################################################
# Sub-Projects
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(wrapper)