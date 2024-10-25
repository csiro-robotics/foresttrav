// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz

#ifndef NDT_APPERANCE_RAY_FUSER_H
#define NDT_APPERANCE_RAY_FUSER_H

#include <ohm/OccupancyMap.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>


namespace nve_core
{
/*** @brief Fuser apperance features into an OHM map
 *
 *  Requires the point to be structured in fashion <ts, global_pos, feature(s)>
 *
 */

enum ColourFusionMode
{
  EndPoint = 0,
  ES = 1,
  Ndt = 2
};

class NdtApperanceRayFuser
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NdtApperanceRayFuser() = default;

  void integrate_rgb_colour(ohm::OccupancyMap &map, cv::Mat &img, const Eigen::Vector3d &sensor_org,
                            const std::vector<Eigen::Vector3d> &points_m, const Eigen::Affine3d &T_c_m);

  bool set_camera_matrix(const std::vector<double> &camera_K_coeff, const std::vector<double> &dist_d_coeff);

  std::vector<Eigen::Vector3d> get_endpoints() { return endpoints_; };
  std::vector<Eigen::Vector3d> get_endpoints_colour() { return endpoints_colour_; };

  void setColouFusionMode(ColourFusionMode mode, double beta);

  ColourFusionMode mode_{};  // Mode for raycasting 0: epoint model 1:
  double beta_{};

  /// Set the mask
  void set_image_mask(std::vector<double> img_bounds)
  {
    img_bounds_ = img_bounds; 
  };
private:
  /// @brief projects a 3d point to a 2d pixel coordinates based on the pinhole camera
  /// @param[in] pi_c Point i in camera frame
  // @param[out] px, py pixel cooordinates 
  void project_point(const Eigen::Vector3d &pi_c, int &px, int &py);

  /// @brief: Fuses a colour with external weights at a given point
  /// @param[in] map Map containning all the data
  /// @param[in] p_m Eendpoint in static world frame [m]
  /// @param[in] rgb_colour Colour in RGB space [0-255] or [0-1]
  /// @param[in] external_weight Weight 
  void endpoint_rgb_fusion(ohm::OccupancyMap &map, const Eigen::Vector3d &p_m, const Eigen::Vector3d &rgb_colour,
                           const double external_weight);

  /// @overload Where the  endpoint is in @p ohm::Key format
  void endpoint_rgb_fusion(ohm::OccupancyMap &map, const ohm::Key &key, const Eigen::Vector3d &rgb_colour,
                           const double external_weight);

  Eigen::Vector3d ray_casting_fusion(ohm::OccupancyMap &map, const Eigen::Vector3d &ray_org,
                                     const Eigen::Vector3d &p_m);

  void ndt_ray_fuser(ohm::OccupancyMap &map, const Eigen::Vector3d &sensor_org, const Eigen::Vector3d &ray_endpoint,
                     const Eigen::Vector3d rgb_colour);

  /// @brief: Ealy stopping ray casting fusion using the ndt hit and miss metric
  Eigen::Vector3d ndt_es_fuser(ohm::OccupancyMap &map, const Eigen::Vector3d &sensor_org,
                               const Eigen::Vector3d &ray_endpoint);

  cv::Mat K_{};  // 3x3 matrix of camera
  cv::Mat d_{};  // 5x1 distortion coefficients

  std::vector<Eigen::Vector3d> endpoints_{};         // Debug to project the visible points/endpoints used
  std::vector<Eigen::Vector3d> endpoints_colour_{};  // Colour associated with the endpoints
  std::vector<double> img_bounds_{0.5, 0.95, 0.0, 1.0};
};


}  // namespace nve_ia


#endif
