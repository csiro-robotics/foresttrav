# Copyright (c) 2022
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author: Fabio Ruetz (fabio.ruetz@csiro.au)
cmake_minimum_required(VERSION 3.10)
project(pdag_cpp)

## Use C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Wpedantic)

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    sensor_msgs
    tf2_ros
    nve_tools
    nve_core
)

## Find system libraries
find_package(Eigen3 REQUIRED)
find_package(ohm REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${EIGEN3_INCLUDE_DIR}
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    tf2_ros
    ohm_ros
    nve_core
)

set(SOURCES
  src/PoseCloudExtractorNode.cpp
)

set(PUBLIC_HEADERS
  src/PoseCloudExtractorNode.h
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OHM_INCLUDE_DIRS}
  # Set manually because Eigen sets a non standard INCLUDE DIR
  ${EIGEN3_INCLUDE_DIR}
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
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ohm::ohm  ${catkin_LIBRARIES})

## Declare cpp executables
add_executable(pose_cloud_extactor_node src/PoseCloudExtractorNode.cpp)
target_link_libraries(pose_cloud_extactor_node ${PROJECT_NAME} )


# Mark executables and/or libraries for installation
install(
  TARGETS   ${PROJECT_NAME} pose_cloud_extactor_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)