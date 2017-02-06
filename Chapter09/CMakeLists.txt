# cmake for OpenCV Cookbook 3rd edition chapter 09
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( patches patches.cpp)
add_executable( matcher matcher.cpp)
add_executable( binaryDescriptors binaryDescriptors.cpp)

# link libraries
target_link_libraries( patches ${OpenCV_LIBS})
target_link_libraries( matcher ${OpenCV_LIBS})
target_link_libraries( binaryDescriptors ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/church01.jpg 
            ${CMAKE_SOURCE_DIR}/images/church02.jpg
            ${CMAKE_SOURCE_DIR}/images/church03.jpg)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
