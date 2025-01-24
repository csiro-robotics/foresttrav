# Copyright (c) 2022
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
#
# Author: Fabio Ruetz (fabio.ruetz@csiro.au)
cmake_minimum_required(VERSION 3.10)
project(collision_mapper)

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
    nve_robot_labelling
    pcl_ros
    pcl_conversions
)

## Find system libraries
# find_package(glm REQUIRED) #External
find_package(Eigen3 REQUIRED)
find_package(ohm REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include
    ${CMAKE_CURRENT_LIST_DIR}/include
    ## This is only necessary because Eigen3 sets a non-standard EIGEN3_INCLUDE_DIR variable
    ${EIGEN3_INCLUDE_DIR}
    ## This is only necessary because Eigen3 sets a non-standard OHM variable
    ${OHM_INCLUDE_DIRS}
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    tf2_ros
    nve_robot_labelling
    pcl_ros
    pcl_conversions
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  # Set manually because OHM sets a non standard INCLUDE DIR
  ${OHM_INCLUDE_DIRS}
  # Set manually because Eigen sets a non standard INCLUDE DIR
  ${EIGEN3_INCLUDE_DIR}
)


## Define the source files
set(SOURCES 
    src/collision_mapper_node.cpp
)


# Add libraries
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
target_link_libraries(${PROJECT_NAME} ohm::ohm  Eigen3::Eigen ${catkin_LIBRARIES})


## Declare cpp executables
add_executable(collision_mapper_node src/collision_mapper_node.cpp)
target_link_libraries(collision_mapper_node ${PROJECT_NAME} )


#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS   ${PROJECT_NAME} collision_mapper_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
## Add launch file directory if applicaple
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

## Add config directory if applicaple
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

## If public library needs to be installed
# install(FILES ${PUBLIC_HEADERS} DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
