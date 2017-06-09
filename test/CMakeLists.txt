cmake_minimum_required(VERSION 3.6)
project(BlenderTest)

###############################################
# Compiler and Linker options
set(CMAKE_CXX_STANDARD 11) 

###############################################
# dependency files
include_directories(../src/wrapper/);

###############################################
# For OpenCV header files
include_directories(/path/to/OpenCV/include);
# For OpenCV libs
link_directories(/path/to/OpenCV/libs);
set(OPENCV_LIBS opencv.libs);

###############################################
set(SOURCE_FILES
    main.cpp)
	
###############################################
# Output result
add_executable(${PROJECT_NAME} ${SOURCE_FILES});
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBS} Blender);