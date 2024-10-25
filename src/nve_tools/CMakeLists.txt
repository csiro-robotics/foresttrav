cmake_minimum_required(VERSION 3.10)
project(nve_tools)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-Wall -Wpedantic)

# find dependencies
find_package(catkin REQUIRED)
find_package(glm REQUIRED) #External

catkin_package(
  INCLUDE_DIRS
    include
    ${CMAKE_CURRENT_LIST_DIR}/include
    ## This is only necessary because Eigen3 sets a non-standard EIGEN3_INCLUDE_DIR variable
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
)

set(SOURCES 
  src/TrajectoryLoader.cpp
  src/ImageTSLoader.cpp
  src/SemanticDataLoader.cpp
)
set(PUBLIC_HEADERS
  include/nve_tools/ImageTSLoader.h
  include/nve_tools/SemanticDataLoader.h
  include/nve_tools/TrajectoryLoader.h
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
target_link_libraries(${PROJECT_NAME} glm::glm ${catkin_LIBRARIES})


install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
