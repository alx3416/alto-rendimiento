# OpenCV_DIR
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
set (OpenCV_USE_STATIC_LIBS ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
include_directories(${OpenCV_INCLUDE_DIRS})


set(TARGET_NAME cpp_opencv)

add_executable(${TARGET_NAME} show_image.cpp
        image_utils.h)

target_link_libraries(cpp_opencv
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)

# Path to your image (relative to this CMakeLists.txt file)
set(SOURCE_IMAGE ${CMAKE_CURRENT_SOURCE_DIR}/baboon.png)

# Path to output directory (where the executable will be)
set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Add a post-build step to copy the image
add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SOURCE_IMAGE}
        ${OUTPUT_DIR}/baboon.png
        COMMENT "Copying baboon.png to output directory"
)