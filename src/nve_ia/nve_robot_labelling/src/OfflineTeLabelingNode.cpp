#include "OfflineVoxelLablerNode.h"
#include "ros/ros.h"


/// TODO
/// - Visualize the poses in map frame for all collision states


int main(int argc, char **argv)
{
  ros::init(argc, argv, "OfflineVoxelTeLabler");
  ros::NodeHandle nh{};
  ros::NodeHandle private_nh{ "~" };
  ros::spinOnce();

  nve_lfe::OfflineVoxelLablerNode labler{nh, private_nh};
  labler.process();

  // Spin ros
  ros::Rate r(1.0);
  while (nh.ok() and not labler.is_pipeline())
  {
    ros::spinOnce();
    r.sleep();
  }


  return 0;
}