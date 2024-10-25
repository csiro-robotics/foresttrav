cmake_minimum_required(VERSION 3.10)
project(nve_startup)

find_package(catkin REQUIRED)
catkin_package()

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)