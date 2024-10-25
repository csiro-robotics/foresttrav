#ifndef LFE_UTILS_H
#define LFE_UTILS_H

#include "NveLabler.h"

#include <Eigen/Core>
#include <vector>

namespace nve_lfe
{

/// @brief Checks if label is one of the binary-labels (Te, NTe)
bool isValidBinaryLabel(const TeLabel label);

/// @brief Checks if it is one of the intialized labels (Te,NTe, Unknown)
bool isInitializedLabel(const TeLabel label);

/// Fast and easy way to generate a discrete set of points which should intersect with all the voxel within the robot
/// bounding box
/// @param[in] robot_bounding_box_points AABB for the robot
/// @param[out] res  spacial resolutions of the points (should be lower than voxel)
/// @param[out] robot_collision_points AABB for collision (front and beyond)
auto generatePointsInsideRobot(const std::vector<double> &bounding_box, double res) -> std::vector<Eigen::Vector3d>;

/// @brief Calculates the log-odds of a probability
auto logsOdds(double prob) -> double;

/// @brief Calculates the probability of a log-odds
auto probFromLogsOdds(double logs_odds) -> double;

/// @brief Returns the label of the voxel with given thresholds
auto getTeLabel(const ohm::SemanticLabel &voxel_data, double te_threshold, double nte_threshold) -> TeLabel;


}  // namespace nve_lfe


#endif  // Header Guard