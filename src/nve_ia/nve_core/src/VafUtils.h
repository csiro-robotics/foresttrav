#ifndef VAF_UTILS
#define VAF_UTILS

#include <cv_bridge/cv_bridge.h>

#include <ohm/Key.h>
#include <ohm/OccupancyMap.h>

#include <Eigen/Core>

namespace nve_core
{
/// @brief: Helper function which checks the 2d point is within image bounds
bool valid2dPoint(const cv::Point2d &uv_point, const double img_width, const double img_height);

bool valid2dPoint(const cv::Point2d &uv_point, const double img_width_min, const double img_width_max,
                  const double img_height_min, const double img_height_max);

/// Helper function to generate the Camera matrix and distortion array
void to_camera_matrix(cv::Mat &K, cv::Mat &d, const std::vector<double> &camera_K_coeff,
                      const std::vector<double> &dist_d_coeff);

std::vector<ohm::Key> unique_keys(const ohm::OccupancyMap &map, const std::vector<Eigen::Vector3d> &points);

std::vector<Eigen::Vector3d> unique_voxel_points(ohm::OccupancyMap &map,
                                                  const std::vector<Eigen::Vector3d> &points);


}  // namespace nve_ia
#endif