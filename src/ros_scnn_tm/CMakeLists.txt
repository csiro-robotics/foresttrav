cmake_minimum_required(VERSION 3.10)
project(ros_scnn_tm)

## Use C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Wpedantic)

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
    ros_numpy
    scnn_tm
    message_generation
    std_msgs
)

# Only use if there is a setup.py
# catkin_python_setup()
add_message_files( DIRECTORY msg FILES   StringArray.msg TriggerModelUpdate.msg OnlineDataSet.msg)

# add_service_files(DIRECTORY srv FILES TriggerWithFile.srv TriggerModelLoad.srv)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS message_runtime rospy  
)

# Mark launch for instalation
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
# Mark launch for confirguration file
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS 
  src/scnn_ftm_ros.py 
  src/scnn_online_trainer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
