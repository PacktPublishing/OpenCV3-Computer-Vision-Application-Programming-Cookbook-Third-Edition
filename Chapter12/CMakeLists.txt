# cmake for OpenCV Cookbook 3rd edition chapter 12
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( videoprocessing videoprocessing.cpp)
add_executable( foreground foreground.cpp)

# link libraries
target_link_libraries( videoprocessing ${OpenCV_LIBS})
target_link_libraries( foreground ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/bike.avi)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
