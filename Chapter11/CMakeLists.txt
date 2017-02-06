# cmake for OpenCV Cookbook 3rd edition chapter 11
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( calibrate calibrate.cpp CameraCalibrator.cpp)
add_executable( cameraPose cameraPose.cpp)
add_executable( estimateE estimateE.cpp triangulate.cpp)
add_executable( stereoMatcher stereoMatcher.cpp)

# link libraries
target_link_libraries( calibrate ${OpenCV_LIBS})
target_link_libraries( cameraPose ${OpenCV_LIBS})
target_link_libraries( estimateE ${OpenCV_LIBS})
target_link_libraries( stereoMatcher ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/bench2.jpg
            ${CMAKE_SOURCE_DIR}/images/soup1.jpg
            ${CMAKE_SOURCE_DIR}/images/soup2.jpg
            ${CMAKE_SOURCE_DIR}/images/brebeuf1.jpg
            ${CMAKE_SOURCE_DIR}/images/brebeuf2.jpg			
            calib.xml)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
FILE(COPY ${CMAKE_SOURCE_DIR}/images/chessboards/ DESTINATION ./chessboards/)
FILE(COPY ${CMAKE_SOURCE_DIR}/images/chessboards/ DESTINATION "Debug/chessboards/")
FILE(COPY ${CMAKE_SOURCE_DIR}/images/chessboards/ DESTINATION "Release/chessboards/")
