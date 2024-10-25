#ifndef BINARY_TE_LABLER_ROS_H
#define BINARY_TE_LABLER_ROS_H

#include "BinaryTeLabler.h"
#include "ros/ros.h"


#include <set>
#include <string>

namespace nve_lfe
{


class BinaryTeLablerROS : public BinaryTeLabler
{
public:
  BinaryTeLablerROS(ros::NodeHandle &nh, ros::NodeHandle &nh_private, std::shared_ptr<ohm::OccupancyMap> map)
    : nh_(nh)
    , nh_private_(nh_private)
  {
    loadRosParams();
    configure(map, params_);
  }

  auto loadRosParams() -> void
  {
    nh_private_.param("override_nte", params_.override_nte, false);
    nh_private_.param("te_probability_threshold", params_.te_probability_threshold, 0.85);
    nh_private_.param("nte_probability_threshold", params_.nte_probability_threshold, 0.15);
    nh_private_.param("collision_bbox", params_.col_box, std::vector<double>{ 0, 0, 0, 0, 0, 0 });
    nh_private_.param("robot_bbox", params_.r_box, std::vector<double>{ 0, 0, 0, 0, 0, 0 });
  }


private:
  ros::NodeHandle nh_, nh_private_;

};  // class BinaryTeLablerROS

}  // namespace nve_lfe
#endif  // Header Guard