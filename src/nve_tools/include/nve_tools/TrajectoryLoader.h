// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
//
// Original author: Thomas Lowe
// Adapted by: Fabio Ruetz
#ifndef NVE_TRAJECTORY_LOADER_H
#define NVE_TRAJECTORY_LOADER_H

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <iostream>
#include <vector>

namespace nve_tools
{
class TrajectoryIO
{
public:
  std::vector<glm::dvec3> &points() { return points_; }

  std::vector<double> &times() { return times_; }

  std::vector<glm::dvec4> &quats() { return quats_; }

  std::vector<int> &labels() { return labels_; }

  // Sets functions for this method. Ugly but functional
  void setPoints(std::vector<glm::dvec3> points) { points_ = points; };
  void setQuats(std::vector<glm::dvec4> quats) { quats_ = quats; };
  void setTimes(std::vector<double> times) { times_ = times; };
  void setLabels(std::vector<int> labels) { labels_ = labels; };

  /// Load trajectory from file. The file is expected to be a text file, with one Node entry per line
  bool load(const std::string &file_name);

  // Writes the trajectory in a new file
  bool write(std::string file_name, bool skip = false);

  /// Nearest pose from a time stamp. Interpolates if not exact. If out of bound will return false
  /// and be clamped to the beginning or end
  size_t nearestPose(double time, glm::dvec3 &output_pos, glm::dvec4 &output_quat);

  // Retrieves nearest pose and returns the label as well. Uses the index of the parent function
  size_t nearestPose(double time, glm::dvec3 &output_pos, glm::dvec4 &output_quat, int &label);

  // Assigne a lable to a timestamp
  void assignLabelToPose(double time, int label);


  // Check if labels have been loaded
  bool has_labels() { return !labels_.empty(); };

private:
  /// Assume constant time spacing!
  /// Returns the lower of the two time bounds
  inline size_t getIndexAndNormaliseTime(double &time) const
  {
    size_t index = std::lower_bound(times_.begin(), times_.end(), time) - times_.begin();
    if (index == 0)
    {
      index++;
    }
    else if (index == times_.size())
    {
      // return index
    }
    index--;

    time = (time - times_[index]) / (times_[index + 1] - times_[index]);
    return index;
  }

  std::vector<glm::dvec3> points_;  // points [m]
  std::vector<double> times_;       // times [s]
  std::vector<glm::dvec4> quats_;   // quaternions 
  std::vector<int> labels_;         // labels  [ ]
};

}  // namespace nve_tools
#endif