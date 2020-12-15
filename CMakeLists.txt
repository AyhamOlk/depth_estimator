CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED PATHS "/usr/lib/x86_64-linux-gnu/cmake/opencv4")

PROJECT(INF573-Project)

ADD_EXECUTABLE(approximate_F approximateFundementalMatrix.cpp)
ADD_EXECUTABLE(disparity disparity.cpp image.cpp headers/image.h)

include_directories(headers)
TARGET_LINK_LIBRARIES(approximate_F ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(disparity ${OpenCV_LIBS})
