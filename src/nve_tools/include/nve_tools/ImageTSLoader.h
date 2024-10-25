// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
//
// Original author: Thomas Lowe
// Adapted by: Fabio Ruetz
#ifndef NVE_IMAGE_TS_LOADER_H
#define NVE_IMAGE_TS_LOADER_H


#include <iostream>
#include <vector>

namespace nve_tools
{
class ImageTSLoader
{
public:
  std::vector<double> &times() { return times_; };
  std::vector<double> times_dc() { return times_; };

  /// Load trajectory from file. The file is expected to be a text file, with one Node entry per line
  bool load(const std::string &file_name);

  /// Nearest index from a time stamp
  size_t nearestFrame(double time);

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

  std::vector<double> times_;  // times [s]
};

}  // namespace nve_tools
#endif