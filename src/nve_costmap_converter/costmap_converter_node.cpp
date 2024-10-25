// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz
#include "Ohm2dCostmapGenerator.h"
#include <ros/ros.h>


int main(int argc, char **argv)
{
  ros::init(argc, argv, "GroundMapGeneration");
  ros::NodeHandle nh{};
  ros::NodeHandle private_nh{ "~" };

  nve_ros::Ohm2dCostmapGenerator costmap_generator(nh, private_nh);
  ros::spin();

  return 0;
}
