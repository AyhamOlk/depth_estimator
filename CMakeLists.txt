CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED PATHS "/usr/lib/x86_64-linux-gnu/cmake/opencv4")

PROJECT(TP3)

ADD_EXECUTABLE(depth_map main.cpp)
ADD_EXECUTABLE(approximate_F approximateFundementalMatrix.cpp)
TARGET_LINK_LIBRARIES(depth_map ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(approximate_F ${OpenCV_LIBS})
