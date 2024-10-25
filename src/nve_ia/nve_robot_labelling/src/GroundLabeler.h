#ifndef GROUND_LABELER_H
#define GROUND_LABELER_H

#include "NveLabler.h"
#include "lfe_utils.h"
#include "nve_core/Types.h"

#include <nve_tools/TrajectoryLoader.h>

#include <ohm/ohm/KeyHash.h>
#include <ohm/ohm/OccupancyMap.h>
#include <ohm/ohm/Voxel.h>
#include <ohm/ohm/VoxelSemanticLabel.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>

#include <set>
#include <string>

namespace nve_lfe
{

/// TODO: Cleanup for interface such that does not retain old crap

struct GroundLabelerParams
{
  double te_update{ 0.6 };
  double nte_update{ 0.4 };
  double te_probability_threshold{ 0.7 };
  double nte_probability_threshold{ 0.2 };

  std::vector<double> collision_bbox{ 0, 0, 0, 0, 0, 0 };
  std::vector<double> robot_bbox{ 0, 0, 0, 0, 0, 0 };
};


/// Note: We are using collision as ground and non-collision (traversable) as non-ground!
class GroundLabler : public Labler
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GroundLabler();
  ~GroundLabler() override = default;

  /// @brief  Sets the occupancy map
  /// @param[in] occupancy_map The occupancy map to be used
  /// @param[in] params The parameters of the labler
  auto configure(std::shared_ptr<ohm::OccupancyMap> map, GroundLabelerParams params) -> void
  {
    map_ = map;
    semantic_layer_ = ohm::Voxel<ohm::SemanticLabel>(map_.get(), map_->layout().semanticLayer());
    params_ = params;
    r_above_ = generatePointsInsideRobot(params_.robot_bbox, map_->resolution());
    r_below_ = generatePointsInsideRobot(params_.collision_bbox, map_->resolution());
  };

  ///
  void processEvent(const Eigen::Affine3d &T_mr, const TeLabel label, const double *prob,
                    const bool *override) override;


  void occupancyUpdate(ohm::SemanticLabel &voxel_label, TeLabel observed_label);

  /// @brief Adds a mapt prior to the map. It is based on the map_te_prior and map_nte_prior_params updates
  /// @param[in] key
  /// @param[in] label
  auto addMapPrior(const ohm::Key key, const TeLabel label) -> void;


  /// @brief: Accessor for a semantic data, it is a hard copy
  /// @param[in]    key         Key to access the voxel data
  /// @param[out]   voxel_data  Is set to zeros if no data is found, values otherwise
  /// @return   True if data is found, false otherwies
  auto getSemanticData(const ohm::Key &key, ohm::SemanticLabel &semantic_label) -> bool override;

protected:
  std::shared_ptr<ohm::OccupancyMap> map_;
  GroundLabelerParams params_{};

  /// Point discretization for the bounding box [x_min, y_min, z_min, x_max, y_max, z_max] in robot frame
  std::vector<Eigen::Vector3d> r_above_{};  // AABB for collision (front and beyond)
  std::vector<Eigen::Vector3d> r_below_{};  /// AABB for the robot

  ohm::Voxel<ohm::SemanticLabel> semantic_layer_;
};


class GroundLablerROS : public GroundLabler
{
public:
  GroundLablerROS(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private, std::shared_ptr<ohm::OccupancyMap> map)
    : nh_(nh)
    , nh_private_(nh_private)
  {
    loadRosParams();
    configure(map, params_);
  }

  auto loadRosParams() -> void
  {
    nh_private_.param("non_ground_update", params_.te_update, 0.6);
    nh_private_.param("ground_update", params_.nte_update, 0.4);
    nh_private_.param("non_ground_threshold", params_.te_probability_threshold, 0.65);
    nh_private_.param("ground_threshold", params_.nte_probability_threshold, 0.35);
    nh_private_.param("ground_collision_bbox", params_.collision_bbox, std::vector<double>{ 0, 0, 0, 0, 0, 0 });
    nh_private_.param("non_ground_bbox", params_.robot_bbox, std::vector<double>{ 0, 0, 0, 0, 0, 0 });


    ROS_INFO_STREAM("non_ground_update: " << params_.te_update);
    ROS_INFO_STREAM("ground_update: " << params_.nte_update);
    ROS_INFO_STREAM("non_ground_threshold: " << params_.te_probability_threshold);
    ROS_INFO_STREAM("ground_threshold: " << params_.nte_probability_threshold);
  }


private:
  const ros::NodeHandle nh_, nh_private_;

};  // class GroundLablerROS

}  // namespace nve_lfe

#endif  // Header Guard