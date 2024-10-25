#include "OhmColourFuserRos.h"
#include "std_msgs/String.h"

#include <sstream>
#include "ros/ros.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "ColourFuserNode");
  ros::NodeHandle nh{};
  ros::NodeHandle private_nh{ "~" };

  nve_ros::OhmColourFuserRos colour_fuser_node(nh, private_nh);

  ros::spin();

  return 0;
}