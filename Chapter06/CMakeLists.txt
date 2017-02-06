# cmake for OpenCV Cookbook 3rd edition chapter 06
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( derivatives derivatives.cpp)
add_executable( filters filters.cpp)

# link libraries
target_link_libraries( derivatives ${OpenCV_LIBS})
target_link_libraries( filters ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/boldt.jpg ${CMAKE_SOURCE_DIR}/images/salted.bmp)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")

