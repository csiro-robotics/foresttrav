
#include "NdtApperanceRayFuser.h"
#include "VafUtils.h"

#include <glm/vec3.hpp>

#include <ohm/DefaultLayer.h>
#include <ohm/Voxel.h>
#include <ohm/VoxelAppearance.h>
#include <ohm/VoxelData.h>
#include <ohm/VoxelSemanticLabel.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <numeric>

namespace nve_core
{
bool NdtApperanceRayFuser::set_camera_matrix(const std::vector<double> &camera_K_coeff,
                                             const std::vector<double> &dist_d_coeff)
{
  if (not camera_K_coeff.empty() and not dist_d_coeff.empty())
  {
    to_camera_matrix(K_, d_, camera_K_coeff, dist_d_coeff);
    return true;
  }
  return false;
}

void NdtApperanceRayFuser::setColouFusionMode(ColourFusionMode mode, double beta)
{
  mode_ = mode;
  beta_ = beta;
}


void NdtApperanceRayFuser::integrate_rgb_colour(ohm::OccupancyMap &map, cv::Mat &img, const Eigen::Vector3d &sensor_org,
                                                const std::vector<Eigen::Vector3d> &points_m,
                                                const Eigen::Affine3d &T_c_m)
{
  /// Visible points
  std::vector<cv::Point3d> cv_p3_c;
  cv_p3_c.reserve(points_m.size());

  for (auto &pi : points_m)
  {
    cv_p3_c.emplace_back(cv::Point3d(pi.x(), pi.y(), pi.z()));
  }

  auto cv_p2 = std::vector<cv::Point2d>();
  cv_p2.reserve(cv_p3_c.size());

  // /// TODO: Remove hacky stuff
  Eigen::Matrix<double, 3, 3> R = T_c_m.rotation();
  Eigen::Matrix<double, 3, 1> t = T_c_m.translation();
  cv::Mat R_cv, t_cv;
  cv::eigen2cv(R, R_cv);
  cv::eigen2cv(t, t_cv);


  cv::projectPoints(cv_p3_c, R_cv, t_cv, K_, d_, cv_p2);

  /// Colours are stored in RGB with range [0,1]
  auto img_width = img.size().width;
  auto img_height = img.size().height;

  auto min_dist = 0.5;
  auto max_dist = 30.0;

  // Padding for image
  auto img_w_low = img_bounds_[0];
  auto img_w_high = img_bounds_[1];
  auto img_h_low = img_bounds_[2];
  auto img_h_high = img_bounds_[3];

  endpoints_.clear();
  endpoints_.reserve(cv_p2.size());
  endpoints_colour_.clear();
  endpoints_colour_.reserve(cv_p2.size());

  /// SHould we undistort the image?
  auto idx_point = 0;
  for (auto &p2d : cv_p2)
  {
    /// Check if it is a valid cv2 point ....
    if (valid2dPoint(p2d, img_width * img_w_low, img_width * img_w_high, img_height * img_h_low,
                     img_height * img_h_high))
    { 
      cv::Vec3b bgr = img.at<cv::Vec3b>(p2d);
      auto rgb_d3 = Eigen::Vector3d(static_cast<double>(bgr[2]) / 255.0, static_cast<double>(bgr[1]) / 255.0,
                                    static_cast<double>(bgr[0]) / 255.0);

      if(ColourFusionMode::EndPoint == mode_)
      {   
        // Simply use all the endpoints
        endpoints_.push_back(points_m[idx_point]);
        endpoints_colour_.push_back(rgb_d3);
        endpoint_rgb_fusion(map, points_m[idx_point], rgb_d3, beta_);
      }
      else if (ColourFusionMode::ES == mode_)
      {
        //Check if the
        auto pi = ray_casting_fusion(map, sensor_org, points_m[idx_point]);
        auto dist = (pi - sensor_org).norm();
        if (min_dist < dist and dist <= (points_m[idx_point] - sensor_org).norm() and dist < max_dist)
        {
          // TODO: think about resizing images
          endpoints_.push_back(pi);
          endpoints_colour_.push_back(rgb_d3);
          endpoint_rgb_fusion(map, pi, rgb_d3, beta_);
        }
      }
      else
      {
        exit(1);
      }
      // else if (ColourFusionMode::Ndt == mode_)
      // {
      //   auto pi = ndt_es_fuser(map, sensor_org, points_m[idx_point]);
      //   auto dist = (pi - sensor_org).norm();
      //   if (min_dist < dist and dist <= (points_m[idx_point] - sensor_org).norm() and dist < max_dist)
      //   {
      //     // TODO: think about resizing images
      //     endpoints_.push_back(pi);
      //     endpoints_colour_.push_back(rgb_d3);
      //     endpoint_rgb_fusion(map, pi, rgb_d3, beta_);
      //   }
      // }
    }
    /// DONT COMMENT THIS OUT
    idx_point++;
  }

  // // /// Debug, we vizualize the pixels...
  if (true)
  {
    cv::Mat img_debug;
    size_t idx = 0;
    for (auto &point_uv : cv_p2)
    {
      if (valid2dPoint(point_uv, img_width * img_w_low, img_width * img_w_high, img_height * img_h_low,
                       img_height * img_h_high))
      {
        cv::circle(img, point_uv, 3, (255, 0, 0), 2);
      }
    }
  }
}


Eigen::Vector3d NdtApperanceRayFuser::ray_casting_fusion(ohm::OccupancyMap &map, const Eigen::Vector3d &ray_org,
                                                         const Eigen::Vector3d &p_m)
{
  ///
  Eigen::Vector3d ray_dir = p_m - ray_org;
  auto dist = ray_dir.norm();
  ray_dir.normalize();

  double lambda = 0.03;
  bool walk = true;
  Eigen::Vector3d pi = ray_org;
  auto occ_layer = ohm::Voxel<float>(&map, map.layout().occupancyLayer());
  auto hm_layer = ohm::Voxel<ohm::HitMissCount>(&map, map.layout().hitMissCountLayer());
  auto hm_value = ohm::HitMissCount();

  /// TDDO: Fix for max ray lentgh and define terminal condition more cleanly
  /// Walk along the ray and terminate at a condition
  /// Cond_1: Occupancy
  /// Cond_2: Permeability
  ///
  /// Note: Check that one has found a terminal state and not just exceded the maximum ray distance!
  while (walk)
  {  /// Keep going
    pi += lambda * ray_dir;
    /// Check condition
    /// If occupied?
    const auto key = map.voxelKey(glm::dvec3(pi.x(), pi.y(), pi.z()));
    ohm::setVoxelKey(key, occ_layer, hm_layer);

    if (/*hm_layer.isValid() and */ occ_layer.isValid())
    {
      // hm_layer.read(&hm_value);
      // auto perm =
      //   static_cast<double>(hm_value.miss_count) / static_cast<double>(hm_value.miss_count + hm_value.hit_count);
      /// Permable and smaller than the endpoint
      walk = not ohm::isOccupied(occ_layer) and (dist > (pi - ray_org).norm());

      // Considerations for permeability based on
      // walk = (perm > 0.9) and (dist > (pi - ray_org).norm());
    }
  }

  ///
  return pi;
}


void NdtApperanceRayFuser::endpoint_rgb_fusion(ohm::OccupancyMap &map, const Eigen::Vector3d &p_m,
                                               const Eigen::Vector3d &rgb_colour, const double external_weight)
{
  auto key = map.voxelKey(glm::dvec3(p_m.x(), p_m.y(), p_m.z()));
  endpoint_rgb_fusion(map, key, rgb_colour, external_weight);
}

void NdtApperanceRayFuser::endpoint_rgb_fusion(ohm::OccupancyMap &map, const ohm::Key &key,
                                               const Eigen::Vector3d &rgb_colour, const double external_weight)
{
  ohm::Voxel<ohm::AppearanceVoxel> color_layer(&map, map.layout().appearanceLayer());
  ohm::AppearanceVoxel color_voxel = ohm::AppearanceVoxel();
  ohm::setVoxelKey(key, color_layer);

  /// OHM stuff
  if (color_layer.isValid())
  {
    color_layer.read(&color_voxel);
  }

  color_voxel.count += 1;

  /// Initialization of Apperance Voxels
  if (color_voxel.count == 1)
  {
    color_voxel.red[0] = rgb_colour[0];  // Measurement is mean
    color_voxel.red[1] = 0.;
    color_voxel.green[0] = rgb_colour[1];  // Measurement is mean
    color_voxel.green[1] = 0.5;
    color_voxel.blue[0] = rgb_colour[2];  // Measurement is mean
    color_voxel.blue[1] = 0.5;
  }

  if (external_weight > 0.0)
  {
    ohm::updateAppearanceVoxel(color_voxel.red, rgb_colour[0], color_voxel.count, external_weight);
    ohm::updateAppearanceVoxel(color_voxel.green, rgb_colour[1], color_voxel.count, external_weight);
    ohm::updateAppearanceVoxel(color_voxel.blue, rgb_colour[2], color_voxel.count, external_weight);
  }
  else
  {
    ohm::updateAppearanceVoxel(color_voxel.red, rgb_colour[0], color_voxel.count);
    ohm::updateAppearanceVoxel(color_voxel.green, rgb_colour[1], color_voxel.count);
    ohm::updateAppearanceVoxel(color_voxel.blue, rgb_colour[2], color_voxel.count);
  }

  color_layer.write(color_voxel);
}


Eigen::Vector3d NdtApperanceRayFuser::ndt_es_fuser(ohm::OccupancyMap &map, const Eigen::Vector3d &sensor_org,
                                                   const Eigen::Vector3d &ray_endpoint)
{
  /// We want to show that occuapnyc may not be the right method to stop for a colour
  /// Consider the NDT hit statistic and decide what to do based on that

  Eigen::Vector3d ray_dir = (ray_endpoint - sensor_org);
  auto dist = (ray_endpoint - sensor_org).norm();
  ray_dir.normalize();
  double lambda = 0.015;  /// TODO: Parameter
  Eigen::Vector3d pi = sensor_org + lambda * ray_dir;

  /// Update for the ray part
  ohm::Key last_key = map.voxelKey(glm::dvec3(sensor_org.x(), sensor_org.y(), sensor_org.z()));
  const auto ray_endpoint_key = map.voxelKey(glm::dvec3(ray_endpoint.x(), ray_endpoint.y(), ray_endpoint.z()));
  ohm::Key key_to_update;

  /// NDT-TM variables for loop
  glm::dvec3 sensor = glm::dvec3(sensor_org.x(), sensor_org.y(), sensor_org.z());
  glm::dvec3 sample = glm::dvec3(ray_endpoint.x(), ray_endpoint.y(), ray_endpoint.z());
  double p_x_ml_given_voxel;
  double p_x_ml_given_sample;
  auto pass_threshold = 0.35;  /// TODO: find out what this value is!!!!!
  auto sensor_noise = 0.05;    /// TODO: Find out what values we are using!

  /// Ohm values
  auto cov_layer = ohm::Voxel<ohm::CovarianceVoxel>(&map, map.layout().covarianceLayer());
  ohm::CovarianceVoxel cov_voxel;
  auto mean_layer = ohm::Voxel<ohm::VoxelMean>(&map, map.layout().meanLayer());
  ohm::VoxelMean mean_voxel;

  while ((pi - sensor_org).norm() < dist and last_key != ray_endpoint_key)
  {
    /// Ray based update to assess wheter we want to update the voxel with a certain weight
    auto key = map.voxelKey(glm::dvec3(pi.x(), pi.y(), pi.z()));
    ohm::setVoxelKey(key, cov_layer, mean_layer);

    if ((last_key != key) and cov_layer.isValid() and mean_layer.isValid())
    {
      cov_layer.read(&cov_voxel);
      mean_layer.read(&mean_voxel);

      /// TODO: Magic number which contains how many information we have per-voxel
      if (mean_voxel.count > 3)
      {
        auto mean_pos = ohm::positionUnsafe(mean_layer);
        auto x_ml = ohm::calculateSampleLikelihoods(&cov_voxel, sensor, sample, mean_pos, sensor_noise,
                                                    &p_x_ml_given_voxel, &p_x_ml_given_sample);
        if (key == ray_endpoint_key)
        {
          // Check for condition C and D
          if ((p_x_ml_given_voxel * p_x_ml_given_sample) >= pass_threshold or p_x_ml_given_voxel >= pass_threshold)
          {
            return pi;
          }
        }
        else
        {
          // Pass trough condition (Case A)
          // We "hit" a distribution
          if (p_x_ml_given_voxel * (1.0 - p_x_ml_given_sample) >= pass_threshold)
          {
            /// Do we need an additional stopping criterion?
            return pi;
          }
        }
      }
    }

    /// Update that should happen at every round of iteration
    last_key = key;
    pi += (lambda * ray_dir);
  }
  return Eigen::Vector3d(0, 0, 0);
}

void NdtApperanceRayFuser::ndt_ray_fuser(ohm::OccupancyMap &map, const Eigen::Vector3d &sensor_org,
                                         const Eigen::Vector3d &ray_endpoint, const Eigen::Vector3d rgb_colour)
{
  /// TODO: Fix the ndt ray fuser for es model
  /// We want to show that occuapnyc may not be the right method to stop for a colour
  /// Consider the NDT hit statistic and decide what to do based on that
  /// Weighted updates are also an option but unclear how that would work....
  Eigen::Vector3d ray_dir = (ray_endpoint - sensor_org);
  auto dist = (ray_endpoint - sensor_org).norm();
  ray_dir.normalize();
  double lambda = 0.015;
  Eigen::Vector3d pi = sensor_org + lambda * ray_dir;

  /// Update for the ray part
  ohm::Key last_key = map.voxelKey(glm::dvec3(sensor_org.x(), sensor_org.y(), sensor_org.z()));
  const auto ray_endpoint_key = map.voxelKey(glm::dvec3(ray_endpoint.x(), ray_endpoint.y(), ray_endpoint.z()));
  std::vector<ohm::Key> keys;
  std::vector<double> p_x_ml_given_voxels;

  /// NDT-TM variables
  glm::dvec3 sensor = glm::dvec3(sensor_org.x(), sensor_org.y(), sensor_org.z());
  glm::dvec3 sample = glm::dvec3(ray_endpoint.x(), ray_endpoint.y(), ray_endpoint.z());
  double p_x_ml_given_voxel;
  double p_x_ml_given_sample;
  auto pass_threshold = 0.35;  /// TODO: find out what this value is!!!!!
  auto sensor_noise = 0.05;    /// TODO: Find out what values we are using!

  /// Ohm values
  auto cov_layer = ohm::Voxel<ohm::CovarianceVoxel>(&map, map.layout().covarianceLayer());
  ohm::CovarianceVoxel cov_voxel;
  auto mean_layer = ohm::Voxel<ohm::VoxelMean>(&map, map.layout().meanLayer());
  ohm::VoxelMean mean_voxel;


  while ((pi - sensor_org).norm() < dist and last_key != ray_endpoint_key)
  {
    /// Ray based update to assess wheter we want to update the voxel with a certain weight
    auto key = map.voxelKey(glm::dvec3(pi.x(), pi.y(), pi.z()));
    ohm::setVoxelKey(key, cov_layer, mean_layer);

    if ((last_key != key) and cov_layer.isValid() and mean_layer.isValid())
    {
      /// We dont want to deal with this case
      cov_layer.read(&cov_voxel);
      mean_layer.read(&mean_voxel);

      /// TODO: Magic number which contains how many information we have per-voxel
      if (mean_voxel.count > 3)
      {
        auto mean_pos = ohm::positionUnsafe(mean_layer);
        auto x_ml = ohm::calculateSampleLikelihoods(&cov_voxel, sensor, sample, mean_pos, sensor_noise,
                                                    &p_x_ml_given_voxel, &p_x_ml_given_sample);

        // Pass trough condition (Case A)
        if (p_x_ml_given_voxel * (1.0 - p_x_ml_given_sample) >= pass_threshold)
        {
          p_x_ml_given_voxels.emplace_back(p_x_ml_given_voxel);
          keys.emplace_back(key);

          endpoints_.emplace_back(Eigen::Vector3d(mean_pos.x, mean_pos.y, mean_pos.z));
          endpoints_colour_.emplace_back(rgb_colour);
        }
        /// Else we do nothing
      }
    }

    /// Update that should happen at every round of iteration
    last_key = key;
    pi += (lambda * ray_dir);
  }

  // Check endpoint
  /// Question:
  if (true)
  {
    ohm::setVoxelKey(ray_endpoint_key, cov_layer, mean_layer);
    cov_layer.read(&cov_voxel);
    mean_layer.read(&mean_voxel);

    /// TODO: Magic number which contains how many information we have per-voxel
    if (mean_voxel.count > 5)
    {
      auto mean_pos = ohm::positionUnsafe(mean_layer);
      auto x_ml = ohm::calculateSampleLikelihoods(&cov_voxel, sensor, sample, mean_pos, sensor_noise,
                                                  &p_x_ml_given_voxel, &p_x_ml_given_sample);

      // End in (Case B-b, C-c) or case D-d
      if ((p_x_ml_given_voxel * p_x_ml_given_sample) >= pass_threshold or p_x_ml_given_voxel >= pass_threshold)
      {
        p_x_ml_given_voxels.emplace_back(p_x_ml_given_voxel);
        keys.emplace_back(ray_endpoint_key);

        endpoints_.emplace_back(Eigen::Vector3d(mean_pos.x, mean_pos.y, mean_pos.z));
        endpoints_colour_.emplace_back(rgb_colour);
      }
    }
  }

  /// Colour fusion strategy
  auto norm_const = 0.0;  // std::accumulate(p_x_ml_given_voxels.begin(), p_x_ml_given_voxels.end(), 0.0);


  /// Softmax weighting

  for (auto &weight : p_x_ml_given_voxels)
  {
    auto log_w = log(weight);
    auto log_w_b = log_w * beta_;
    weight = exp(log(weight) * beta_);
    norm_const += weight;
  }
  auto elements = p_x_ml_given_voxels.size();
  int idx = 0;
  for (auto &weight : p_x_ml_given_voxels)
  {
    endpoint_rgb_fusion(map, keys[idx], rgb_colour, weight);
    idx++;
  }
}
}  // namespace nve_core
