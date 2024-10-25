#include <lfe_coverage_map/LfeCoverageMapNode.h>

#include <chrono>
#include "ros/ros.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "lfe_coverage_node");
  ros::NodeHandle nh{};
  ros::NodeHandle private_nh{ "~" };

  nve_lfe::LfeCoverageNode feature_map_node(nh, private_nh);

  ros::spin();

  return 0;
}