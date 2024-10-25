// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author: Fabio Ruetz (fabio.ruetz@csiro.au)

#ifndef PCL_CAMERA_VISABILITY_CHECKER_H
#define PCL_CAMERA_VISABILITY_CHECKER_H

// https://computergraphics.stackexchange.com/questions/4640/point-respect-to-plane/4642
/// More indept methods...
// http://web.archive.org/web/20120531231005/http://crazyjoke.free.fr/doc/3D/plane%20extraction.pdf

#include <string>
#include <vector>

#include <Eigen/Core>

namespace nve_core
{
class VisabilityChecker
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VisabilityChecker() = default;

  void set_camera_matrix(const std::vector<double> &camera_matrix_params);

  // Check if a single point is inside the frustum
  bool point_inside_frustum(const Eigen::Vector3d &points);

  /// @brief Goes through all points and only returns the ones within the furstum
  std::vector<Eigen::Vector3d> check_points(const std::vector<Eigen::Vector3d> &points_in);

private:
  Eigen::Vector3d n1_, n2_, n3_, n4_;
};

void VisabilityChecker::set_camera_matrix(const std::vector<double> &camera_matrix_params)
{
  Eigen::Matrix<double, 3, 3> K, K_inv;
  K.setZero();
  K(0, 0) = camera_matrix_params[0];  // fx
  K(1, 1) = camera_matrix_params[1];  // fy
  K(0, 2) = camera_matrix_params[2];  // cx
  K(1, 2) = camera_matrix_params[3];  // cy
  K(2, 2) = 1.0;

  K_inv = K.inverse();

  Eigen::Vector3d r_1 = K_inv * Eigen::Vector3d(camera_matrix_params[0] * 2.0, 0.0, 1);
  Eigen::Vector3d r_2 = K_inv * Eigen::Vector3d(camera_matrix_params[0] * 2.0, camera_matrix_params[1] * 2.0, 1);
  Eigen::Vector3d r_3 = K_inv * Eigen::Vector3d(0, camera_matrix_params[1] * 2.0, 1);
  Eigen::Vector3d r_4 = K_inv * Eigen::Vector3d(0, 0, 1);

  n1_ = r_1.cross(r_2);
  n2_ = r_2.cross(r_3);
  n3_ = r_3.cross(r_4);
  n4_ = r_4.cross(r_1);
}

bool VisabilityChecker::point_inside_frustum(const Eigen::Vector3d &points)
{
  bool f1 = n1_.dot(points) > 0.0;
  bool f2 = n2_.dot(points) > 0.0;
  bool f3 = n3_.dot(points) > 0.0;
  bool f4 = n4_.dot(points) > 0.0;

  return f1 and f2 and f3 and f4;
}

std::vector<Eigen::Vector3d> VisabilityChecker::check_points(const std::vector<Eigen::Vector3d> &points_in)
{
  std::vector<Eigen::Vector3d> visible_points;
  visible_points.reserve(points_in.size());

  for (auto &p : points_in)
  {
    if (point_inside_frustum(p))
    {
      visible_points.push_back(p);
    }
  }
  return visible_points;
}

}  // namespace nve_ia


#endif