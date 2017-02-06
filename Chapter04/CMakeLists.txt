# cmake for OpenCV Cookbook 3rd edition chapter 04
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( histograms histograms.cpp)
add_executable( contentfinder contentfinder.cpp)
add_executable( finder finder.cpp)
add_executable( retrieve retrieve.cpp)
add_executable( integral integral.cpp)
add_executable( tracking tracking.cpp)

# link libraries
target_link_libraries( histograms ${OpenCV_LIBS})
target_link_libraries( contentfinder ${OpenCV_LIBS})
target_link_libraries( finder ${OpenCV_LIBS})
target_link_libraries( retrieve ${OpenCV_LIBS})
target_link_libraries( integral ${OpenCV_LIBS})
target_link_libraries( tracking ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/group.jpg 
            ${CMAKE_SOURCE_DIR}/images/waves.jpg 
			${CMAKE_SOURCE_DIR}/images/dog.jpg 
			${CMAKE_SOURCE_DIR}/images/baboon01.jpg 
			${CMAKE_SOURCE_DIR}/images/baboon02.jpg 
			${CMAKE_SOURCE_DIR}/images/beach.jpg 
			${CMAKE_SOURCE_DIR}/images/bear.jpg 
			${CMAKE_SOURCE_DIR}/images/lake.jpg 
			${CMAKE_SOURCE_DIR}/images/moose.jpg 
			${CMAKE_SOURCE_DIR}/images/polar.jpg
			${CMAKE_SOURCE_DIR}/images/marais.jpg
			${CMAKE_SOURCE_DIR}/images/fundy.jpg
			${CMAKE_SOURCE_DIR}/images/book.jpg
			${CMAKE_SOURCE_DIR}/images/bike55.bmp
			${CMAKE_SOURCE_DIR}/images/bike65.bmp)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")