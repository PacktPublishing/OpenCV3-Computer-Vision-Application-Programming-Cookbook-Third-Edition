# cmake for OpenCV Cookbook 3rd edition chapter 05
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( morphology morphology.cpp)
add_executable( segment segment.cpp)
add_executable( mserFeatures mserFeatures.cpp)

# link libraries
target_link_libraries( morphology ${OpenCV_LIBS})
target_link_libraries( segment ${OpenCV_LIBS})
target_link_libraries( mserFeatures ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/binary.bmp 
			${CMAKE_SOURCE_DIR}/images/building.jpg
			${CMAKE_SOURCE_DIR}/images/group.jpg
			${CMAKE_SOURCE_DIR}/images/book.jpg
			${CMAKE_SOURCE_DIR}/images/boldt.jpg
			${CMAKE_SOURCE_DIR}/images/tower.jpg)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")

