#include "ProbTeLabler.h"
#include "lfe_utils.h"

#include <ohm/ohm/CovarianceVoxel.h>
#include <ohm/ohm/Key.h>
#include <ohm/ohm/KeyList.h>
#include <ohm/ohm/KeyRange.h>
#include <ohm/ohm/VoxelSemanticLabel.h>

#include <assert.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace nve_lfe
{

ProbTeLabler::ProbTeLabler()
{}

auto ProbTeLabler::configure(std::shared_ptr<ohm::OccupancyMap> map, ProbTeLabelerParams params) -> void
{
  map_ = map;
  updateParams(params);
}

auto ProbTeLabler::updateParams(ProbTeLabelerParams params) -> void
{
  params_ = params;
  robot_collision_points_ = generatePointsInsideRobot(params_.col_box, map_->resolution());
  robot_bounding_box_points_ = generatePointsInsideRobot(params_.r_box, map_->resolution());
}


auto ProbTeLabler::labelPoseWithTargetLabel(const Eigen::Affine3d &T_mr, TeLabel label) -> void
{
  /// Assumption made that we are using binary TE classificaiton
  if (not(label == TeLabel::Collision or label == TeLabel::Free))
  {
    return;
  }

  auto &roi_points = TeLabel::Free == label ? robot_bounding_box_points_ : robot_collision_points_;

  if (roi_points.size() == 0)
  {
    return;
  }
  auto semantic_layer = ohm::Voxel<ohm::SemanticLabel>(map_.get(), map_->layout().semanticLayer());
  ohm::SemanticLabel voxel_semantics;
  for (auto &pi_r : roi_points)
  {
    auto pi_m = T_mr * pi_r;
    const auto key = map_->voxelKey(glm::dvec3(pi_m.x(), pi_m.y(), pi_m.z()));
    ohm::setVoxelKey(key, semantic_layer);

    if (not semantic_layer.isValid())
    {
      continue;
    }

    semantic_layer.read(&voxel_semantics);

    if (not params_.fixed_map_prior)
    {
      occupancyUpdateWithMapPrior(voxel_semantics, label);
    }
    else
    {
      occupancyUpdate(voxel_semantics, label);
    }

    semantic_layer.write(voxel_semantics);
  }
}

void ProbTeLabler::occupancyUpdate(ohm::SemanticLabel &voxel_label, TeLabel observed_label)
{
  /// Check voxel is not initialized and intializes appropriately
  if (voxel_label.label == TeLabel::Uninitialized)
  {
    /// Assumption that is is equaly likely that  map is occupied ot not
    /// Label needed to not disciriminate
    voxel_label.label = TeLabel::Unknown;
    voxel_label.prob_label = logsOdds(0.5);
  }

  /// TODO: Open question about the probability prior of the voxel being occupied or not.
  if (observed_label == TeLabel::Collision)
  {
    voxel_label.prob_label = voxel_label.prob_label + logsOdds(params_.nte_update) - logsOdds(0.5);
  }
  else if (observed_label == TeLabel::Free)
  {
    voxel_label.prob_label = voxel_label.prob_label + logsOdds(params_.te_update) - logsOdds(0.5);
  }

  voxel_label.label = getTeLabel(voxel_label, params_.te_probability_threshold, params_.nte_probability_threshold);
}


auto ProbTeLabler::occupancyUpdateWithMapPrior(ohm::SemanticLabel &voxel_label, TeLabel label) -> void
{
  if (voxel_label.label == TeLabel::Uninitialized)
  {
    /// Assumption that is is equaly likely that  map is occupied ot not
    /// Label needed to not disciriminate
    voxel_label.label = TeLabel::Unknown;
    voxel_label.prob_label = logsOdds(0.5);
  }

  /// Find out which map prior we use
  double map_prior_prob = 0.5;
  if (TeLabel::Collision == voxel_label.state_label)
  {
    map_prior_prob = params_.nte_prior_prob;
  }
  else if (TeLabel::Free == voxel_label.state_label)
  {
    map_prior_prob = params_.te_prior_prob;
  }

  /// TODO: Open question about the probability prior of the voxel being occupied or not.
  if (label == TeLabel::Collision)
  {
    if (map_prior_prob != 0.5)
    {
      auto b = 1;
    }
    voxel_label.prob_label = voxel_label.prob_label + logsOdds(params_.nte_update) + logsOdds(map_prior_prob)-logsOdds(0.5);
  }
  else if (label == TeLabel::Free)
  {
    voxel_label.prob_label = voxel_label.prob_label + logsOdds(params_.te_update) + logsOdds(map_prior_prob)-logsOdds(0.5);
  }
  voxel_label.label = getTeLabel(voxel_label, params_.te_probability_threshold, params_.nte_probability_threshold);
}

auto ProbTeLabler::addMapPrior(const ohm::Key key, const TeLabel label) -> void
{
  if (not isValidBinaryLabel(label))
  {
    return;
  }

  auto semantic_layer = ohm::Voxel<ohm::SemanticLabel>(map_.get(), map_->layout().semanticLayer());
  ohm::setVoxelKey(key, semantic_layer);

  if (not semantic_layer.isValid())
  {
    return;
  }
  ohm::SemanticLabel voxel_semantics;
  semantic_layer.read(&voxel_semantics);

  /// We fix for all voxel a singular state
  if (params_.fixed_map_prior)
  {
    /// Heuristic to skip the voxel if it has been set to a collision
    if (voxel_semantics.prob_label > logsOdds(0.0001) and
        voxel_semantics.prob_label < logsOdds(params_.nte_prior_prob + 0.05))
    {
      return;
    }

    voxel_semantics.label = TeLabel::Unknown;
    voxel_semantics.state_label = TeLabel::Unknown;
    voxel_semantics.prob_label =
      TeLabel::Free == label ? logsOdds(params_.te_prior_prob) : logsOdds(params_.nte_prior_prob);
  }
  else
  {
    /// Set the state of the voxel to our map prior
    /// We do this since it is not garuanteed that we have more than one update
    voxel_semantics.label = TeLabel::Unknown;
    voxel_semantics.state_label = label;
    voxel_semantics.prob_label =
      TeLabel::Free == label ? logsOdds(params_.te_prior_prob) : logsOdds(params_.nte_prior_prob);
  }

  semantic_layer.write(voxel_semantics);
}


auto ProbTeLabler::getSemanticData(const ohm::Key &key, ohm::SemanticLabel &semantic_label) -> bool
{ 
  auto semantic_layer = ohm::Voxel<ohm::SemanticLabel>(map_.get(), map_->layout().semanticLayer());
  ohm::setVoxelKey(key, semantic_layer);

  semantic_label.label = TeLabel::Uninitialized;
  semantic_label.prob_label = -1.0;

  if (not semantic_layer.isValid())
  {
    return false;
  }

  semantic_layer.read(&semantic_label);

  // Only if the state and semantic label are unitialized should we return false, else the map-prior was set by hand
  if (semantic_label.label == TeLabel::Uninitialized)
  {
    return false;
  }

  semantic_label.label =
    getTeLabel(semantic_label, params_.te_probability_threshold, params_.nte_probability_threshold);
  return true;
}


}  // namespace nve_lfe