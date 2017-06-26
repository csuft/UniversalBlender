cmake_minimum_required(VERSION 3.6)
project(BlenderTest C CXX)

###############################################
# Compiler and Linker options
set(CMAKE_CXX_STANDARD 11) 

###############################################
# dependency files
include_directories(../src/wrapper/) 
if (WIN32)
link_directories(../lib/win/)
set(UNROLLMAP_LIBS UnrollMap64.lib)
elseif (APPLE)
link_directories(../lib/mac/)
set(UNROLLMAP_LIBS libunrollmap.dylib)
endif(WIN32)

###############################################
# Configure OpenCV.
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

###############################################
set(SOURCE_FILES
    main.cpp)
	
###############################################
# Output result
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Blender)