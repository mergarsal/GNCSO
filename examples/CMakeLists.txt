project(GNCSO-Examples CXX)

# Find Eigen library
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")
find_package(Eigen3 3.3.3 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library (version ${EIGEN3_VERSION})")
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}\n")
else()
message(STATUS "Eigen library not found!")
endif()

# Add the Eigen include directories
include_directories(${EIGEN3_INCLUDE_DIR})


# Rotation test: Wahba problem
add_executable(GNC_rotation_example ${CMAKE_CURRENT_SOURCE_DIR}/GNC_rotation_example.cpp)
target_link_libraries(GNC_rotation_example gncso)

# Relpose test: RelPose with essential matrix (DLT solver)
add_executable(GNC_essential_example ${CMAKE_CURRENT_SOURCE_DIR}/GNC_essential_example.cpp)
target_link_libraries(GNC_essential_example gncso)

