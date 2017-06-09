cmake_minimum_required(VERSION 3.6)
project(Blender)

##################################################
# Compiler and Linker options
set(CMAKE_CXX_STANDARD 11)  

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
link_directories(../lib/win/)
set(UNROLLMAP_LIBS UnrollMap64.lib)

###################################################
# Output result
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${UNROLLMAP_LIBS} ${OpenCL_LIBRARIES})