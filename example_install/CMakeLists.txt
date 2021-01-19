project(GNCSO-Examples-Inst CXX)

cmake_minimum_required(VERSION 3.1)
# PROJECT CONFIGURATION
project(gncso LANGUAGES CXX VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 14) # We require C++ 14 or later
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Find Eigen library
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")
find_package(Eigen3 3.3.3 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library (version ${EIGEN3_VERSION})")
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}\n")
else()
message(STATUS "Eigen library not found!")
endif()

# Add the Eigen include directories
include_directories(${EIGEN3_INCLUDE_DIR})


find_package(gncso REQUIRED)

# Rotation test: Wahba problem
add_executable(GNC_rotation_example GNC_rotation_example.cpp)
target_link_libraries(GNC_rotation_example gncso)

