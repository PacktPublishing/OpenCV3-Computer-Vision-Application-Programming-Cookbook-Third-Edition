# cmake for OpenCV Cookbook 3rd edition chapter 01
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( loadDisplaySave loadDisplaySave.cpp)
add_executable( logo logo.cpp)
add_executable( mat mat.cpp)

# link libraries
target_link_libraries( loadDisplaySave ${OpenCV_LIBS})
target_link_libraries( logo ${OpenCV_LIBS})
target_link_libraries( mat ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/puppy.bmp ${CMAKE_SOURCE_DIR}/images/smalllogo.png)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
