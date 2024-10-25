// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
//
// Author: Fabio Ruetz
// Adapted from Tomas Low
#include "nve_tools/ImageTSLoader.h"


#include <assert.h>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace nve_tools
{
bool ImageTSLoader::load(const std::string &file_name)
{
  std::string line;
  int size = 0;

  // Pre-counting the size of the file
  {
    std::ifstream ifs(file_name.c_str(), std::ios::in);
    if (!ifs)
    {
      std::cerr << "(1) Failed to open video time stamp file: " << file_name << std::endl;
      return false;
    }
    assert(ifs.is_open());
    getline(ifs, line);

    while (!ifs.eof())
    {
      getline(ifs, line);
      if (line.empty())
      {
        break;
      }
      size++;
    }
  }

  std::ifstream ifs(file_name.c_str(), std::ios::in);
  if (!ifs)
  {
    std::cerr << "(2) Failed to open  video time stamp file : " << file_name << std::endl;
    return false;
  }
  // getline(ifs, line);  // We expect header to be removed here

  times_.resize(size);

  bool ordered = true;

  std::string time_sec = "";
  std::string time_sub_sec = "";
  for (int i = 0; i < size; i++)
  {
    if (!ifs)
    {
      std::cerr << "Invalid stream when loading trajectory file1: " << file_name << std::endl;
      return false;
    }

    getline(ifs, line);
    std::istringstream iss(line);

    if (line.empty())
    {
      break;
    }

    iss >> time_sec >> time_sub_sec;
    times_[i] = std::stod(time_sec + "." + time_sub_sec);
  }

  return true;
}

size_t ImageTSLoader::nearestFrame(double time)
{
  double ratio = time;
  size_t index = getIndexAndNormaliseTime(ratio);

  index = ratio >= 0.5 ? index + 1 : index;

  if (0 >= index || times_.size())
  {
    return -1;
  }
  return index;
}

}  // namespace nve_tools