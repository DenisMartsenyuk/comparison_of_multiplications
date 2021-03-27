cmake_minimum_required(VERSION 3.17)
project(comparison_of_multiplications)

set(CMAKE_CXX_STANDARD 14)

add_executable(comparison_of_multiplications main.cpp matrices/MatrixOperations.h gpu/MultiplicationMatricesGPU.cpp gpu/MultiplicationMatricesGPU.h config.h)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
target_include_directories (comparison_of_multiplications PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (comparison_of_multiplications ${OpenCL_LIBRARY})