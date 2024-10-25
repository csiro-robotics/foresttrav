#include "GroundLabeler.h"
#include "lfe_utils.h"

namespace nve_lfe
{

GroundLabler::GroundLabler()
{}

auto GroundLabler::processEvent(const Eigen::Affine3d &T_mr, const TeLabel label, const double *prob,
                                const bool *override) -> void
{
  /// We update the occupancy and free at the same time. Each pose has a free and a collision update
  ohm::SemanticLabel voxel_semantics;
  for (auto &pi_r : r_above_)
  {
    auto pi_m = T_mr * pi_r;
    const auto key = map_->voxelKey(glm::dvec3(pi_m.x(), pi_m.y(), pi_m.z()));
    ohm::setVoxelKey(key, semantic_layer_);

    if (not semantic_layer_.isValid())
    {
      continue;
    }

    semantic_layer_.read(&voxel_semantics);
    occupancyUpdate(voxel_semantics, nve_lfe::TeLabel::Free);
    semantic_layer_.write(voxel_semantics);
  }


  /// Do the collision (ground) update
  for (auto &pi_r : r_below_)
  {
    auto pi_m = T_mr * pi_r;
    const auto key = map_->voxelKey(glm::dvec3(pi_m.x(), pi_m.y(), pi_m.z()));
    ohm::setVoxelKey(key, semantic_layer_);

    if (not semantic_layer_.isValid())
    {
      continue;
    }

    semantic_layer_.read(&voxel_semantics);
    occupancyUpdate(voxel_semantics, nve_lfe::TeLabel::Collision);
    semantic_layer_.write(voxel_semantics);
  }
}


void GroundLabler::occupancyUpdate(ohm::SemanticLabel &voxel_label, TeLabel observed_label)
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

auto GroundLabler::addMapPrior(const ohm::Key key, const TeLabel label) -> void
{
  return;
}


/// TODO: This is a copy for ProbTeLabler.cpp:
auto GroundLabler::getSemanticData(const ohm::Key &key, ohm::SemanticLabel &semantic_label) -> bool
{
  ohm::setVoxelKey(key, semantic_layer_);

  semantic_label.label = TeLabel::Uninitialized;
  semantic_label.prob_label = -1.0;

  if (not semantic_layer_.isValid())
  {
    return false;
  }

  semantic_layer_.read(&semantic_label);

  if (semantic_label.label == TeLabel::Uninitialized)
  {
    return false;
  }

  semantic_label.label = getTeLabel(semantic_label, params_.te_probability_threshold, params_.nte_probability_threshold);

  return true;
}


}  // namespace nve_lfe