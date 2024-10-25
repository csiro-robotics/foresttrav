// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
//
// Author: Fabio Ruetz
// Adapted from Tomas Low
#include "nve_tools/SemanticDataLoader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>


namespace nve_tools
{

bool SemanticDataLoader::load(const std::string &file_name)
{
  std::cout << "[Semantic Data Loader] Loading semantic data " << file_name << std::endl;
  std::string line;
  int size = 0;
  auto has_prob = false;
  // Pre-counting the size of the file
  {
    std::ifstream ifs(file_name.c_str(), std::ios::in);
    if (!ifs)
    {
      std::cerr << "1) Failed to open trajectory file: " << file_name << std::endl;
      return false;
    }
    assert(ifs.is_open());
    getline(ifs, line);
    if (line.find("prob_1") != std::string::npos)
    {
      has_prob = true;
    }

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

  pos_.resize(size);
  labels_.resize(size);
  prob_.resize(size);

  std::ifstream ifs(file_name.c_str(), std::ios::in);
  if (!ifs)
  {
    std::cerr << "2) Failed to open trajectory file: " << file_name << std::endl;
    return false;
  }
  getline(ifs, line);  // We expect header to be removed here


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

    std::vector<std::string> tokens;
    std::string token;

    while (std::getline(iss, token, ','))
    {
      tokens.push_back(token);
    }

    /// Need to understand how this works!
    if (tokens.size() > 0)
    {
      labels_[i] = std::stoi(tokens[0]);
      pos_[i][0] = std::stod(tokens[1]);
      pos_[i][1] = std::stod(tokens[2]);
      pos_[i][2] = std::stod(tokens[3]);

      if (has_prob)
      {
        prob_[i] = 1000.0 * std::stod(tokens[5]);
      }
    }
  }

  return true;
}

}  // namespace nve_tools