cmake_minimum_required(VERSION 3.6)
project(Blender C CXX)

##################################################
# Compiler and Linker options
set(CMAKE_CXX_STANDARD 11)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS)
##################################################
# Sources and Headers
set(SOURCE_FILES
    base/BaseBlender.h
	base/BaseBlender.cpp
	cpu/CPUBlender.h
	cpu/CPUBlender.cpp
	opencl/OpenCLBlender.h
	opencl/OpenCLBlender.cpp
	utils/base64.h
	utils/base64.cpp
	utils/log.h
	utils/log.cpp
	utils/timer.h
	utils/timer.cpp
	utils/UnrollMap.h
	wrapper/BlenderWrapper.h
	wrapper/BlenderWrapper.cpp)

###################################################
# OpenCL
find_package(OpenCL REQUIRED)

if(OpenCL_FOUND)
	include_directories(${OpenCL_INCLUDE_DIRS})
endif(OpenCL_FOUND)

###################################################
# dependency files
if (WIN32)
link_directories(../lib/win/)
set(UNROLLMAP_LIBS UnrollMap64.lib)
elseif (APPLE)
link_directories(../lib/mac/)
set(UNROLLMAP_LIBS libunrollmap.dylib)
endif(WIN32)

###################################################
# Output result
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${UNROLLMAP_LIBS} ${OpenCL_LIBRARIES})
