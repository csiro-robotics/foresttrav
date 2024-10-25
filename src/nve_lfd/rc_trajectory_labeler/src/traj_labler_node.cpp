#include "TrajLablerROS.h"
#include "std_msgs/String.h"

#include <sstream>
#include "ros/ros.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "TrajLablerROS");
  ros::NodeHandle nh{};
  ros::NodeHandle private_nh{ "~" };

  nve_lfe::TrajectoryLablerNode labler(nh, private_nh);

  ros::spin();

  return 0;
}
