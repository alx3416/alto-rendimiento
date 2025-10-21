cmake_minimum_required(VERSION 4.0)
project(alto_rendimiento)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hello_world main.cpp)

add_subdirectory(basics)
add_subdirectory(opencv)
add_subdirectory(openmp)
add_subdirectory(eigen_basics)
add_subdirectory(img_proc)
