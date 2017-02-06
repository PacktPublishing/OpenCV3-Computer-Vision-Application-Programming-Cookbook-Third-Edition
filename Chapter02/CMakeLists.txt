# cmake for OpenCV Cookbook 3rd edition chapter 02
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( saltImage saltImage.cpp)
add_executable( colorReduce colorReduce.cpp)
add_executable( contrast contrast.cpp)
add_executable( addImages addImages.cpp)
add_executable( remapping remapping.cpp)

# link libraries
target_link_libraries( saltImage ${OpenCV_LIBS})
target_link_libraries( colorReduce ${OpenCV_LIBS})
target_link_libraries( contrast ${OpenCV_LIBS})
target_link_libraries( addImages ${OpenCV_LIBS})
target_link_libraries( remapping ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/boldt.jpg ${CMAKE_SOURCE_DIR}/images/rain.jpg)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")