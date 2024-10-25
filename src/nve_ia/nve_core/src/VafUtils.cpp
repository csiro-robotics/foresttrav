#include "VafUtils.h"

#include <unordered_set>
#include <ohm/Key.h>
#include <ohm/Voxel.h>
#include <ohm/VoxelMean.h>
#include <ohm/DefaultLayer.h>

namespace nve_core
{
/// @brief: Helper function which checks the 2d point is within image bounds
bool valid2dPoint(const cv::Point2d &uv_point, const double img_width, const double img_height)
{
  /// We add 2 pixel padding
  return static_cast<bool>(0 <= uv_point.x && img_width - 2 > uv_point.x && img_height - 2 > uv_point.y &&
                           0 <= uv_point.y);
}

bool valid2dPoint(const cv::Point2d &uv_point, const double img_width_min, const double img_width_max,
                  const double img_height_min, const double img_height_max)
{
  // auto low_end = img_width_min + 2.0 <= uv_point.x and uv_point.x <= img_width_max - 2.0;
  // auto up_end = (img_height_min + 2.0 <= uv_point.y and uv_point.y <= img_height_max - 2.0);
  return (img_width_min + 2.0 <= uv_point.x and uv_point.x <= img_width_max - 2.0) and
         (img_height_min + 2.0 <= uv_point.y and uv_point.y <= img_height_max - 2.0);
}

void to_camera_matrix(cv::Mat &K, cv::Mat &d, const std::vector<double> &camera_K_coeff,
                      const std::vector<double> &dist_d_coeff)
{
  K = (cv::Mat_<double>(3, 3) << camera_K_coeff[0], 0.0, camera_K_coeff[2], 0.0, camera_K_coeff[1], camera_K_coeff[3],
       0.0, 0.0, 1.0);
  d = (cv::Mat_<double>(5, 1) << dist_d_coeff[0], dist_d_coeff[1], dist_d_coeff[2], dist_d_coeff[3], dist_d_coeff[4]);
}

std::vector<ohm::Key> unique_keys(const ohm::OccupancyMap &map, const std::vector<Eigen::Vector3d> &points)
{

  std::unordered_set<ohm::Key> key_set;

  for (auto &p : points)
  {
    const ohm::Key key = map.voxelKey(glm::dvec3(p.x(), p.y(), p.z()));
    if (key_set.find(key) != key_set.end())
    {
      key_set.insert(key);
    }
  }
  std::vector<ohm::Key> unique_keys(key_set.begin(), key_set.end());
  return unique_keys;
}


std::vector<Eigen::Vector3d> unique_voxel_points(ohm::OccupancyMap &map, const std::vector<Eigen::Vector3d> &points)
{
  std::unordered_set<ohm::Key> key_set;

  std::vector<Eigen::Vector3d> unique_points;
  unique_points.reserve(points.size());
  ohm::Voxel<ohm::VoxelMean> mean_layer(&map, map.layout().meanLayer());
  ohm::VoxelMean mean = ohm::VoxelMean();

  for (auto &p : points)
  {
    const ohm::Key key = map.voxelKey(glm::dvec3(p.x(), p.y(), p.z()));
    if (key_set.find(key) != key_set.end())
    { 
      key_set.insert(key);
      ohm::setVoxelKey(key, mean_layer);
      auto pos = ohm::positionSafe(mean_layer);
      unique_points.emplace_back(Eigen::Vector3d(pos.x, pos.y, pos.z));
    }
  }

  unique_points.shrink_to_fit();
  return unique_points;
}

}  // namespace nve_core
