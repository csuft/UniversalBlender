cmake_minimum_required(VERSION 3.6)
project(BlenderWrapper)

set(CMAKE_CXX_STANDARD 11) 

include_directories(../src/wrapper);

set(SOURCE_FILES
    Wrapper.cpp
	Wrapper.h)

####################################################
# Output result
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Blender);