# cmake for OpenCV Cookbook 3rd edition chapter 13
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( tracker tracker.cpp)
add_executable( flow flow.cpp)
add_executable( oTracker oTracker.cpp)

# link libraries
target_link_libraries( tracker ${OpenCV_LIBS})
target_link_libraries( flow ${OpenCV_LIBS})
target_link_libraries( oTracker ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/bike.avi)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
FILE(COPY ${CMAKE_SOURCE_DIR}/images/goose/ DESTINATION ./goose/)
FILE(COPY ${CMAKE_SOURCE_DIR}/images/goose/ DESTINATION "Debug/goose/")
FILE(COPY ${CMAKE_SOURCE_DIR}/images/goose/ DESTINATION "Release/goose/")

