cmake_minimum_required(VERSION 3.1) 
project(yolo-trt)            
enable_language(CUDA)

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

#cuda
#export PATH=/usr/local/cuda-11.0/bin:$PATH
#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)

#opencv
#set(OpenCV_DIR /opt/opencv-4.1.2/share/OpenCV)
find_package(OpenCV REQUIRED)

#detector lib (cuda)
file(GLOB_RECURSE sources modules/*.hpp modules/*.cpp modules/*.h modules/*.cu extra/*.h)
add_library(detector SHARED ${sources})
target_include_directories(detector PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(detector nvinfer nvinfer_plugin nvparsers "stdc++fs")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

find_package(Ceres REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

catkin_package()

#sample
add_executable(sample samples/sample_detector.cpp)				  
target_include_directories(sample PRIVATE modules/ extra/)
target_link_libraries(sample detector ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})


#signal
add_executable(signal samples/signal_detector.cpp)				  
target_include_directories(signal PRIVATE modules/ extra/)
target_link_libraries(signal detector ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

