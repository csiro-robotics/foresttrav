# Copyright (c) 2022
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author: Fabio Ruetz (fabio.ruetz@csiro.au)
cmake_minimum_required(VERSION 3.10)
project(nve_robot_labelling)

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
    nve_core
    geometry_msgs
    pcl_ros
    pcl_conversions
    nve_tools
)


## Find system libraries
find_package(ohm REQUIRED)
find_package(Eigen3 REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${EIGEN3_INCLUDE_DIR}
    ${OHM_INCLUDE_DIRS}
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    tf2_ros
    geometry_msgs
    pcl_ros
    pcl_conversions
    nve_core
    nve_tools
  # DEPENDS

)

set(SOURCES
  src/ProbTeLabler.cpp
  src/BinaryTeLabler.cpp
  src/GroundLabeler.cpp
  src/lfe_utils.cpp
  # src/OfflineVoxelLablerNode.cpp
)

set(PUBLIC_HEADERS
  src/NveLabler.h
  src/ProbTeLabler.h
  src/ProbTeLablerROS.h
  src/BinaryTeLabler.h
  src/BinaryTeLablerROS.h
  src/tinycolourmap.h
  src/lfe_utils.h
  # src/OfflineVoxelLablerNode.h
  src/GroundLabeler.h
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
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ohm::ohm ohm::slamio  ${catkin_LIBRARIES})


## Declare cpp executables
# add_executable(offline_te_labeler_node src/OfflineTeLabelingNode.cpp)
# target_link_libraries(offline_te_labeler_node ${PROJECT_NAME} )


# Mark executables and/or libraries for installation
install(
  TARGETS   ${PROJECT_NAME} 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
