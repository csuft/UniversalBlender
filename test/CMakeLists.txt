cmake_minimum_required(VERSION 3.6)
project(BlenderTest)

###############################################
# Compiler and Linker options
set(CMAKE_CXX_STANDARD 11) 

###############################################
# dependency files
include_directories(../src/wrapper/)

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