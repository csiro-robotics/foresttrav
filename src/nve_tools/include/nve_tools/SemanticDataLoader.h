// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
//
// Original author: Thomas Lowe
// Adapted by: Fabio Ruetz
#ifndef SEMANTIC_DATA_LOADER_H
#define SEMANTIC_DATA_LOADER_H

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <iostream>
#include <vector>

namespace nve_tools
{

/// @brief Loads semantic data generated in the nve_methods
class SemanticDataLoader
{
public:
  std::vector<glm::dvec3> &pos() { return pos_; }
  std::vector<int> &labels() { return labels_; }
  std::vector<int> &probailities() { return prob_; };

  /// Load trajectory from file. The file is expected to be a text file, with one Node entry per line
  bool load(const std::string &file_name);

private:
  /// Assume constant time spacing!
  /// Returns the lower of the two time bounds

  std::vector<glm::dvec3> pos_;  // position of points [m]
  std::vector<int> labels_;      // labels[ rads?]
  std::vector<int> prob_;
};


}  // namespace nve_tools
#endif