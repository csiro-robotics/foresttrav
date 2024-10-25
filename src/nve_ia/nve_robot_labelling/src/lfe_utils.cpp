#include "lfe_utils.h"

namespace nve_lfe
{

bool isValidBinaryLabel(const TeLabel label)
{
  return (label == TeLabel::Collision or label == TeLabel::Free);
}

bool isInitializedLabel(const TeLabel label)
{
  return (label == TeLabel::Collision or label == TeLabel::Free or label == TeLabel::Unknown);
}

auto generatePointsInsideRobot(const std::vector<double> &bounding_box, double res) -> std::vector<Eigen::Vector3d>
{
  std::vector<Eigen::Vector3d> points;
  points.reserve(std::abs(bounding_box[3] - bounding_box[0]) / res + std::abs(bounding_box[4] - bounding_box[1]) / res +
                 std::abs(bounding_box[5] - bounding_box[1]) / res);

  for (double x_i = bounding_box[0]; x_i <= bounding_box[3]; x_i += res)
  {
    for (double y_i = bounding_box[1]; y_i <= bounding_box[4]; y_i += res)
    {
      for (double z_i = bounding_box[2]; z_i <= bounding_box[5]; z_i += res)
      {
        points.emplace_back(Eigen::Vector3d(x_i, y_i, z_i));
      }
    }
  }
  points.shrink_to_fit();

  return points;
}

auto logsOdds(double prob) -> double
{
  return std::log(prob / (1 - prob));
}

auto probFromLogsOdds(double logs_odds) -> double
{
  return std::exp(logs_odds) / (1 + std::exp(logs_odds));
}

auto getTeLabel(const ohm::SemanticLabel &voxel_data, double te_threshold, double nte_threshold) -> TeLabel
{
  /// Return if unknown
  if (nte_threshold <= probFromLogsOdds(voxel_data.prob_label) &&
      probFromLogsOdds(voxel_data.prob_label) < te_threshold)
  {
    return TeLabel::Unknown;
  }

  return probFromLogsOdds(voxel_data.prob_label) < nte_threshold ? TeLabel::Collision : TeLabel::Free;
}

}  // namespace nve_lfe
