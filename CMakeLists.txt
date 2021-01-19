# mgarsal 11/04/2020
# A simple CMakeLists file that enables the Optimization library to be conveniently imported by other CMake projects using
cmake_minimum_required(VERSION 3.1)

# PROJECT CONFIGURATION
project(gncso LANGUAGES CXX VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 14) # We require C++ 14 or later
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to 'RelWithDebInfo' if one was not specified by the user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS RelWithDebInfo Release Debug MinSizeRel)
  message(STATUS "Setting build type to ${CMAKE_BUILD_TYPE}, as none was specified\n")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode\n")
endif()

# BUILD CONFIGURATIONS
# Build the examples?
set(BUILD_GNCSO_EXAMPLES ON CACHE BOOL "Build example executables? [disabled by default]")
# Enable faster instruction sets (SIMD/AVX)
set(ENABLE_VECTORIZATION OFF CACHE BOOL "Enable vectorized instruction sets (SIMD/AVX)? [disabled by default]")

message(STATUS "Processing CMakeLists for header-only library ${PROJECT_NAME}:\n")



## provide us with a set of variables containing installation directories for various artifacts
include(GNUInstallDirs)


# SET UP EXPORT OF GNCSO LIBRARY

# We declare this library as an INTERFACE library; this means that it does not directly produce built output, though it may have properties set on it and it may be installed, exported and imported.
add_library(${PROJECT_NAME} INTERFACE)


# Set the include directory for this project.  We use include/ as the top-level include directory so that #include directives in client source files are of the form "${PROJECT_NAME}/blah/blah.h"
set(GNCSO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
message(STATUS "Found ${PROJECT_NAME} include directory: ${GNCSO_INCLUDE_DIR}")


# The target path for installation can be set by setting the CMAKE_INSTALL_PREFIX variable.
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${GNCSO_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)



# Top-level directory of the GNCSO library
set(GNCSO_TOPLEVEL_DIR ${GNCSO_INCLUDE_DIR}/gncso)

# The list of header files provided by this project
set(GNCSO_HDRS
    ${GNCSO_TOPLEVEL_DIR}/Refinement/Expansion/ExpansionInliers.h
    ${GNCSO_TOPLEVEL_DIR}/Refinement/Base/Concepts.h
    ${GNCSO_TOPLEVEL_DIR}/Base/BaseConcepts.h
    ${GNCSO_TOPLEVEL_DIR}/Base/Concepts.h
    ${GNCSO_TOPLEVEL_DIR}/Base/GNC.h
    ${GNCSO_TOPLEVEL_DIR}/Smooth/gnc_smooth.h
    ${GNCSO_TOPLEVEL_DIR}/GM/GM_fcn.h
    ${GNCSO_TOPLEVEL_DIR}/TLS/TLS_fcn.h
    ${GNCSO_TOPLEVEL_DIR}/Welsch/Welsch.h
    ${GNCSO_TOPLEVEL_DIR}/Tukey/Tukey.h
    )
message(STATUS "Found ${PROJECT_NAME} header files:\n ${GNCSO_HDRS}")


# CMAKECONFIG_INSTALL_DIR must not be absolute paths.
set(LIBRARY_TARGET_NAME ${PROJECT_NAME})
set(include_install_dir "include/${PROJECT_NAME}")
set(LIB_INSTALL_DIR "${LIBRARY_TARGET_NAME}")
set(CMAKECONFIG_INSTALL_DIR "share/${LIBRARY_TARGET_NAME}")
set(RELATIVE_CMAKECONFIG_INSTALL_DIR "share/${LIBRARY_TARGET_NAME}")
set(config_install_dir "lib/cmake/${LIBRARY_TARGET_NAME}-${PROJECT_VERSION}")
set(exported_targets_name "${LIBRARY_TARGET_NAME}Targets")
set(exported_targets_filename "${exported_targets_name}.cmake")
set(export_dirpath "lib/cmake/${LIBRARY_TARGET_NAME}")
set(config_basename "${LIBRARY_TARGET_NAME}Config")
set(config_filename "${config_basename}.cmake")
set(version_filename "${config_basename}Version.cmake")




include(CMakePackageConfigHelpers)
        
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


write_basic_package_version_file("${version_filename}"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)
                                 
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME} DESTINATION include)                               
                       
                       
# Create a TheiaConfig.cmake file. <name>Config.cmake files are searched by
# FIND_PACKAGE() automatically. We configure that file so that we can put any
# information we want in it, e.g. version numbers, include directories, etc.
configure_package_config_file(
                                "${CMAKE_SOURCE_DIR}/cmake/${config_filename}.in"
                                "${CMAKE_CURRENT_BINARY_DIR}/${config_filename}"
                                INSTALL_DESTINATION "${config_install_dir}")
                                
 # List the complete set of header files as source files.
target_sources(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${GNCSO_INCLUDE_DIR}>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
                                
 
                       
# Install these files into the same directory as the generated exports-file,
# we include the FindPackage scripts for libraries whose headers are included
# in the public API of Theia and should thus be present in THEIA_INCLUDE_DIRS.
                 
install(FILES "${PROJECT_BINARY_DIR}/${config_filename}"
              "${PROJECT_BINARY_DIR}/${version_filename}"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
                               
    

# Add add entry for this project into CMake's package registry, so that this project can be found by other CMake projects
export(PACKAGE ${PROJECT_NAME})



  

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
        
        
        
        
# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeUninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/CMakeUninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake/CMakeUninstall.cmake)
endif()



# SET UP [OPTIONAL] CODE EXAMPLES

# Define some additional (optional) named targets, not built by default.  Typically, these will be e.g. unit tests, example executables.
# This is the name of the target in the generated Makefile that builds the example executables; i.e., to build these, one must type "make $EXAMPLES_TARGET_NAME" using the generated Makefile in order to build the example executables (we do this because we don't want these executables to be automatically built by other CMake projects that use this one as a dependency)
if(${BUILD_GNCSO_EXAMPLES})
        message(STATUS "Adding examples to build")
        add_subdirectory(examples)
endif()



