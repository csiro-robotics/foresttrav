#ifndef BINARY_TE_LABLER_H
#define BINARY_TE_LABLER_H

#include "NveLabler.h"
#include "lfe_utils.h"

#include <ohm/ohm/KeyHash.h>
#include <ohm/ohm/OccupancyMap.h>
#include <ohm/ohm/Voxel.h>
#include <ohm/ohm/VoxelSemanticLabel.h>

namespace nve_lfe
{

struct BinaryTeLablerParams
{
  bool override_nte{ false };
  double te_probability_threshold{ 0.85 };
  double nte_probability_threshold{ 0.15 };
  std::vector<double> col_box{ 0, 0, 0, 0, 0, 0 };
  std::vector<double> r_box{ 0, 0, 0, 0, 0, 0 };
};

/** @brief The BinaryTeLabler used poses of a trajectory with a collision event to label an @c ohm::Map
 * as either unknown, collision, free
 *
 * Assumptions:
 *    - A cell is one of the states, not   
 * 
 * @param
 *  override_nt:                Allows to override non-traversable labels with traversable ones
 * te_probability_threshold:    Thershold for which a cell is considered traversable, < 0.5
 * nte_probability_threshold:   Thershold for which a cell is considered non-traversable > 0.5
 * col_box:                     Collision boundary box which is used detect the collision between the argent and the environemnt
 * r_box:                       Robot bounding box which is used for the traversable_update
 * 
 */
class BinaryTeLabler : public Labler
{


public:
  BinaryTeLabler();
  ~BinaryTeLabler() override = default;

  /// @brief  Sets the occupancy map
  /// @param[in] occupancy_map The occupancy map to be used
  /// @param[in] params The parameters of the labler
  auto configure(std::shared_ptr<ohm::OccupancyMap> map, BinaryTeLablerParams params) -> void
  {
    map_ = map;
    semantic_layer_ = ohm::Voxel<ohm::SemanticLabel>(map_.get(), map_->layout().semanticLayer());
    params_ = params;
    robot_collision_points_ = generatePointsInsideRobot(params_.col_box, map_->resolution());
    robot_bounding_box_points_ = generatePointsInsideRobot(params_.r_box, map_->resolution());
  };

  auto processEvent(const Eigen::Affine3d &T_mr, const TeLabel label, const double *prob, const bool *override)
    -> void override;

  /// Adds Map priors
  /// Note: Sets them just above/below the thresholds
  auto addMapPrior(const ohm::Key key, const TeLabel label) -> void override;

  ///@brief Update for binary prefrence for Non-traversable labels
  /// @param[in] key Map key
  /// @param[in] label TeLabel passed to this method. Returns if it is a non-valid (Te, NTe) label
  auto binaryNTePrefrenceUpdate(const ohm::Key key, const TeLabel label) -> void;


  /// @brief: Accessor for a semantic data, it is a hard copy
  /// @param[in]    key         Key to access the voxel data
  /// @param[out]   voxel_data  Is set to zeros if no data is found, values otherwise
  /// @return   True if data is found, false otherwies
  auto getSemanticData(const ohm::Key &key, ohm::SemanticLabel &semantic_label) -> bool override;

protected:
  std::shared_ptr<ohm::OccupancyMap> map_;
  BinaryTeLablerParams params_{};

  /// Point discretization for the bounding box [x_min, y_min, z_min, x_max, y_max, z_max] in robot frame
  std::vector<Eigen::Vector3d> robot_collision_points_{};     // AABB for collision (front and beyond)
  std::vector<Eigen::Vector3d> robot_bounding_box_points_{};  /// AABB for the robot

  ohm::Voxel<ohm::SemanticLabel> semantic_layer_;
};

}  // namespace nve_lfe
#endif  // BINARY_TE_LABLER_H
