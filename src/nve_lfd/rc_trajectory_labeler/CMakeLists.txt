# Copyright (c) 2022
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author: Fabio Ruetz (fabio.ruetz@csiro.au)
cmake_minimum_required(VERSION 3.10)
project(rc_trajectory_labeler)

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
    visualization_msgs
    nve_tools
    rosbag
)


## Find system libraries
find_package(Eigen3 REQUIRED)

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
    visualization_msgs
    nve_tools
    rosbag
)

set(SOURCES
  src/TrajLablerROS.cpp
)

set(PUBLIC_HEADERS
  src/TrajLablerROS.h
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


add_library(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  include
  ${catkin_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}  ${catkin_LIBRARIES})


## Declare cpp executables
add_executable(traj_labler_node src/traj_labler_node.cpp)
target_link_libraries(traj_labler_node ${PROJECT_NAME} )


# Mark executables and/or libraries for installation
install(
  TARGETS   ${PROJECT_NAME} traj_labler_node
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