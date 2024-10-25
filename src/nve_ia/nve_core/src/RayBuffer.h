// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author: Fabio Ruetz (fabio.ruetz@csiro.au)

#ifndef RAY_BUFFER_H
#define RAY_BUFFER_H

#include <slamio/PointCloudReaderPly.h>

#include <string>
#include <vector>

namespace nve_core
{

class RayBuffer
{
public:
  RayBuffer() = default;
  ~RayBuffer();

  bool open_file(std::string file, const double inner_ts, const double outer_ts);

  // Check whether end of file has been reached
  bool end_of_file() { return end_of_file_; };

  /// @brief Gets all points around a timestamp
  bool get_points_around(const double timestamp, std::vector<glm::dvec3> &ray_endpoints);

  void update_buffers();

  bool need_to_update(const double time_stamp);

  void update_around_ts(const double timestamp);

  void remove_non_valid_samples();

  /// Brief the size of how many points are in the buffer
  double buffer_size() { return buffered_points_.size(); };


private:
  double target_timestamp_;
  double inner_ts_; 
  double outer_ts_;

  std::vector<slamio::CloudPoint> buffered_points_;
  std::vector<double> buffered_timestamps_;

  slamio::PointCloudReaderPly cloud_loader_;

  bool end_of_file_;
};


}  // namespace nve_core
#endif