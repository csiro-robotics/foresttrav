#include "BinaryTeLabler.h"
#include "lfe_utils.h"

namespace nve_lfe
{

BinaryTeLabler::BinaryTeLabler()
{}

auto BinaryTeLabler::processEvent(const Eigen::Affine3d &T_mr, const TeLabel label, const double *prob,
                                  const bool *override) -> void
{
  /// Assumption made that we are using binary TE classificaiton
  if (not isValidBinaryLabel(label))
  {
    return;
  }

  auto &roi_points = TeLabel::Free == label ? robot_bounding_box_points_ : robot_collision_points_;

  if (roi_points.size() == 0)
  {
    return;
  }

  ohm::SemanticLabel voxel_semantics;
  for (auto &pi_r : roi_points)
  {
    auto pi_m = T_mr * pi_r;
    const auto key = map_->voxelKey(glm::dvec3(pi_m.x(), pi_m.y(), pi_m.z()));
    ohm::setVoxelKey(key, semantic_layer_);

    if (not semantic_layer_.isValid())
    {
      continue;
    }

    semantic_layer_.read(&voxel_semantics);

    if (TeLabel::Collision == voxel_semantics.label and not params_.override_nte)
    {
      continue;
    }

    voxel_semantics.label = label;
    voxel_semantics.prob_label = TeLabel::Free == label ? logsOdds(0.9999) : logsOdds(0.00001);
    semantic_layer_.write(voxel_semantics);
  }
}

auto BinaryTeLabler::addMapPrior(const ohm::Key key, const TeLabel label) ->void
{
  if (not isValidBinaryLabel(label))
  {
    return;
  }
  ohm::SemanticLabel voxel_semantics;
  ohm::setVoxelKey(key, semantic_layer_);

  if (not semantic_layer_.isValid())
  {
    return;
  }

  semantic_layer_.read(&voxel_semantics);

  /// Check if we assigned a non-traversable label at some point
  if (voxel_semantics.prob_label > logsOdds(0.0001) and
      voxel_semantics.prob_label < logsOdds(params_.nte_probability_threshold))
  {
    return;
  }

  /// Note: The label is set to Unknown know interfere with the update but to be initalized.
  /// Note: We set the labels such that they are just above/below the prob threshold.
  voxel_semantics.label = TeLabel::Unknown;
  voxel_semantics.prob_label = TeLabel::Free == label ? logsOdds(params_.te_probability_threshold + 0.005) :
                                                        logsOdds(params_.nte_probability_threshold - 0.005);
  semantic_layer_.write(voxel_semantics);
}


/// TODO: This is a copy for ProbTeLabler.cpp:
auto BinaryTeLabler::getSemanticData(const ohm::Key &key, ohm::SemanticLabel &semantic_label) -> bool
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

  semantic_label.label =
    getTeLabel(semantic_label, params_.te_probability_threshold, params_.nte_probability_threshold);

  return true;
}


}  // namespace nve_lfe