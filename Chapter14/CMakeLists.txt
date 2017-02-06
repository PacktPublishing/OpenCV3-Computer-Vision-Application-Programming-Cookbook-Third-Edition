# cmake for OpenCV Cookbook 3rd edition chapter 14
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( recognizeFace recognizeFace.cpp)
add_executable( detectObjects detectObjects.cpp)
add_executable( trainSVM trainSVM.cpp)

# link libraries
target_link_libraries( recognizeFace ${OpenCV_LIBS})
target_link_libraries( detectObjects ${OpenCV_LIBS})
target_link_libraries( trainSVM ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/girl.jpg
            ${CMAKE_SOURCE_DIR}/images/face0_1.png
            ${CMAKE_SOURCE_DIR}/images/face0_2.png
            ${CMAKE_SOURCE_DIR}/images/face1_1.png
            ${CMAKE_SOURCE_DIR}/images/face1_2.png
            ${CMAKE_SOURCE_DIR}/images/person.jpg
)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
FILE(COPY ${CMAKE_SOURCE_DIR}/images/stopSamples/ DESTINATION ./stopSamples/)
FILE(COPY ${CMAKE_SOURCE_DIR}/images/stopSamples/ DESTINATION "Debug/stopSamples/")
FILE(COPY ${CMAKE_SOURCE_DIR}/images/stopSamples/ DESTINATION "Release/stopSamples/")


