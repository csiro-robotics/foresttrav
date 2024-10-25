#ifndef PROB_TE_LABELER_H
#define PROB_TE_LABELER_H

#include "NveLabler.h"
#include "nve_core/Types.h"

#include <nve_tools/TrajectoryLoader.h>

#include <ohm/ohm/KeyHash.h>
#include <ohm/ohm/OccupancyMap.h>
#include <ohm/ohm/Voxel.h>
#include <ohm/ohm/VoxelSemanticLabel.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <set>
#include <string>

namespace nve_lfe
{

/// TODO: Cleanup for interface such that does not retain old crap

struct ProbTeLabelerParams
{
  double te_update{ 0.6 };
  double nte_update{ 0.4 };
  double te_probability_threshold{ 0.7 };
  double nte_probability_threshold{ 0.2 };

  /// Map Prior Updatevalues
  bool fixed_map_prior{false};
  double te_prior_prob{0.55};
  double nte_prior_prob{0.45};

  std::vector<double> col_box{ 0, 0, 0, 0, 0, 0 };
  std::vector<double> r_box{ 0, 0, 0, 0, 0, 0 };
};


class ProbTeLabler : public Labler
{
public:
  ProbTeLabler();
  ~ProbTeLabler() override = default;

  ///
  void processEvent(const Eigen::Affine3d &T_mr, const TeLabel label, const double *prob, const bool *override) override
  {
    labelPoseWithTargetLabel(T_mr, label);
  };

  /// @brief  Sets the occupancy map
  /// @param[in] occupancy_map The occupancy map to be used
  /// @param[in] params The parameters of the labler
  auto configure(std::shared_ptr<ohm::OccupancyMap> map, ProbTeLabelerParams params) -> void;

  /// @brief Labels the voxels based on the pose of trajectory
  /// @param[in] pose_traj The trajectory to label the voxels
  /// @param[in] label The label associated with the target pose
  auto labelPoseWithTargetLabel(const Eigen::Affine3d &T_mr, const TeLabel label) -> void;

  /// @brief Occupancy update for a given state
  /// @param[in] voxel_data voxel_data currently stored in the map
  /// @param[in] observed_label Currently observed label
  /// note: The occupancy updates provbalilities are defined in the @p TeLablerParams
  auto occupancyUpdate(ohm::SemanticLabel &voxel_data, TeLabel observed_label) -> void;

  /// @brief Occupancy update for a given state using the map priors from hand labels
  /// @param[in] voxel_data voxel_data currently stored in the map
  /// @param[in] observed_label Currently observed label
  /// note: The occupancy updates provbalilities are defined in the @p TeLablerParams
  auto occupancyUpdateWithMapPrior(ohm::SemanticLabel &voxel_label, TeLabel label)->void;

  /// @brief Adds a mapt prior to the map. It is based on the map_te_prior and map_nte_prior_params updates
  /// @param[in] key
  /// @param[in] label
  auto addMapPrior(const ohm::Key key, const TeLabel label) ->void;

  /// @brief Manual way to set the semantic voxel
  /// Used in two occasions:
  ///   1) When adding hand labellled data with specific labels and probabilities
  ///   2) When adding heuristic labells and specific override functions
  /// @param[in] voxel The voxel to be set
  /// @param[in] label The label to be set to
  /// @param[in] probability The probability to be set
  /// @param[in] override If true, the voxel will be overwritten
  auto setVoxelLabel(const Eigen::Vector3d &voxel_coord, TeLabel label, double proba, bool override) -> void;

  /// @brief: Accessor for a semantic data, it is a hard copy
  /// @param[in]    key         Key to access the voxel data
  /// @param[out]   voxel_data  Is set to zeros if no data is found, values otherwise
  /// @return   True if data is found, false otherwies
  auto getSemanticData(const ohm::Key &key, ohm::SemanticLabel &semantic_label) -> bool override;

  auto updateParams(ProbTeLabelerParams params) -> void;

  std::shared_ptr<ohm::OccupancyMap> map_;
  ProbTeLabelerParams params_{};
  /// Point discretization for the bounding box [x_min, y_min, z_min, x_max, y_max, z_max] in robot frame
protected:
  std::vector<Eigen::Vector3d> robot_collision_points_{};     // AABB for collision (front and beyond)
  std::vector<Eigen::Vector3d> robot_bounding_box_points_{};  /// AABB for the robot

};


}  // namespace nve_lfe

#endif  // Header Guard