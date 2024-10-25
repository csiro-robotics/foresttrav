cmake_minimum_required(VERSION 3.10)
project(lfe_coverage_map)

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
    geometry_msgs
    costmap_2d
    tf2_ros
    pcl_conversions
    pcl_ros
)


## Find system libraries
find_package(Eigen3 REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include
    ${CMAKE_CURRENT_LIST_DIR}/include
    ## This is only necessary because Eigen3 sets a non-standard EIGEN3_INCLUDE_DIR variable
    # ${EIGEN3_INCLUDE_DIR}
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    geometry_msgs
    tf2_ros
    costmap_2d
    pcl_ros
    pcl_conversions

  ## Boost is not part of the DEPENDS since it is only used in source files,
  ## Dependees do not depend on Boost when they depend on this package.
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  # Set manually because Eigen sets a non standard INCLUDE DIR
)

## Declare a C++ library
add_library(${PROJECT_NAME}
   src/LfeCoverageMapNode.cpp
 )
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${catkin_LIBRARIES})

## Declare cpp executables
add_executable(lfe_coverage_map_node src/lfd_coverage_map_node.cpp)
target_link_libraries(lfe_coverage_map_node ${PROJECT_NAME} )


# Mark executables and/or libraries for installation
install(
  TARGETS   lfe_coverage_map_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


#############
## Install ##
#############



# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

# Mark launch for instalation
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

# if(${CATKIN_ENABLE_TESTING})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# ## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test
#   test/test_ros_package_template.cpp
#   test/AlgorithmTest.cpp)
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME}_core)
# endif()

##########################
## Static code analysis ##
##########################

# roslint_cpp()
# catkin_package()