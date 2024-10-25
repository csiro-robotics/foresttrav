#ifndef NVE_LABELER_ROS_H
#define NVE_LABELER_ROS_H

#include "ProbTeLabler.h"
#include "ros/ros.h"

#include <set>
#include <string>

namespace nve_lfe
{


class ProbTeLablerROS : public ProbTeLabler
{
public:
  ProbTeLablerROS(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private, std::shared_ptr<ohm::OccupancyMap> map)
    : nh_(nh)
    , nh_private_(nh_private)
  {
    loadRosParams();
    configure(map, params_);
  }

  auto loadRosParams() -> void
  {
    nh_private_.param("te_update", params_.te_update, 0.6);
    nh_private_.param("nte_update", params_.nte_update, 0.4);
    nh_private_.param("te_probability_threshold", params_.te_probability_threshold, 0.65);
    nh_private_.param("nte_probability_threshold", params_.nte_probability_threshold, 0.35);
    nh_private_.param("fixed_map_prior", params_.fixed_map_prior, true);
    nh_private_.param("te_prior_prob", params_.te_prior_prob, 0.58);
    nh_private_.param("nte_prior_prob", params_.nte_prior_prob, 0.42);
    nh_private_.param("collision_bbox", params_.col_box, std::vector<double>{ 0, 0, 0, 0, 0, 0 });
    nh_private_.param("robot_bbox", params_.r_box, std::vector<double>{ 0, 0, 0, 0, 0, 0 });


    if ((params_.te_prior_prob >= params_.te_update) or
        (params_.nte_prior_prob <= params_.nte_update) and (not params_.fixed_map_prior))
    {
      ROS_WARN_STREAM("Prior updates do not meet the numerical stable conditions p(m_i| TE) > p_(m_i| z_t, x_t) and "
                      "p(m_i| NTE) < p_(m_i| z_t, x_t) ");
    }

    ROS_INFO_STREAM("te_update: " << params_.te_update);
    ROS_INFO_STREAM("nte_update: " << params_.nte_update);
    ROS_INFO_STREAM("te_probability_threshold: " << params_.te_probability_threshold);
    ROS_INFO_STREAM("nte_probability_threshold: " << params_.nte_probability_threshold);
    ROS_INFO_STREAM("fixed_map_prior: " << params_.fixed_map_prior);
    ROS_INFO_STREAM("te_prior_prob: " << params_.te_prior_prob);
    ROS_INFO_STREAM("nte_prior_prob: " << params_.nte_prior_prob);

    /// Check that the bounding boxes are valid
    if ((params_.col_box[3] - params_.col_box[0] <= 0) or (params_.col_box[5] - params_.col_box[2] <= 0) or
        (params_.col_box[4] - params_.col_box[1] <= 0))
    {
      ROS_ERROR_STREAM("Collision bounding box is not valid");
    }

    if ((params_.r_box[3] - params_.r_box[0] <= 0) or (params_.r_box[5] - params_.r_box[2] <= 0) or
        (params_.r_box[4] - params_.r_box[1] <= 0))
    {
      ROS_ERROR_STREAM("Robot bounding box is not valid");
    }
  }


  auto updateMap(std::shared_ptr<ohm::OccupancyMap> map) -> void { configure(map, params_);};

private:
  const ros::NodeHandle nh_, nh_private_;

};  // class ProbTeLablerROS

}  // namespace nve_lfe
#endif  // Header Guard