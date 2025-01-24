# Copyright (c) 2022
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author: Fabio Ruetz (fabio.ruetz@csiro.au)

cmake_minimum_required(VERSION 3.10)
project(nve_core)

set(PACKAGE_VERSION 0.0.0)
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/package.xml")
  file(READ package.xml PACKAGE_XML)
  string(REGEX REPLACE ".*<version>[\\t ]*(.*)[\\t ]*<\\/version>.*" "\\1" PACKAGE_VERSION "${PACKAGE_XML}")
endif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/package.xml")
set("${PROJECT_NAME}_VERSION" "${PACKAGE_VERSION}")

# C++ standards setup.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Ensure -fPIC is added.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # Additional CMake modules
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
## Find catkin macros and libraries
find_package(catkin REQUIRED 
  COMPONENTS
  cv_bridge
)

find_package(ohm REQUIRED)
find_package(Eigen3 REQUIRED)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${EIGEN3_INCLUDE_DIR}
    ${OHM_INCLUDE_DIRS}
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    # cv_bridge
  # DEPENDS
)

set(SOURCES
  # src/ColourFuser.cpp
  src/CsvWriter.cpp
  # src/FeatureExtractor.cpp
  # src/GroundMapInterface.cpp
  # src/NdtApperanceRayFuser.cpp
  # src/OhmVoxelStatistic.cpp
  # src/RayBuffer.cpp
  # src/VafUtils.cpp
)

set(PUBLIC_HEADERS
  src/Types.h
  # src/FeatureExtractor.h
  # src/ColourFuser.h
  # src/RayBuffer.h
  # src/NdtApperanceRayFuser.h
  # src/VafUtils.h
  # src/VisabilityAssesser.h
  # src/GroundMapInterface.h
  src/CsvWriter.h
  # src/OhmVoxelStatistic.h
)

add_library(${PROJECT_NAME} ${SOURCES})

add_dependencies(${PROJECT_NAME} 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  include
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ohm::ohm ohm::slamio ${catkin_LIBRARIES})

# clang_tidy_target(${PROJECT_NAME} EXCLUDE_MATCHES ".*\\.in($|\\..*)")

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
