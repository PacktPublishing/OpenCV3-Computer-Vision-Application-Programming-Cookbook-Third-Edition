# cmake for OpenCV Cookbook 3rd edition chapter 07
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( contours contours.cpp)
add_executable( blobs blobs.cpp)

# link libraries
target_link_libraries( contours ${OpenCV_LIBS})
target_link_libraries( blobs ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/group.jpg 
            ${CMAKE_SOURCE_DIR}/images/binaryGroup.bmp
            ${CMAKE_SOURCE_DIR}/images/mser.bmp
            ${CMAKE_SOURCE_DIR}/images/road.jpg
            ${CMAKE_SOURCE_DIR}/images/chariot.jpg)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
