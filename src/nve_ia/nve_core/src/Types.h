#ifndef NVE_TYPES
#define NVE_TYPES
#include <vector>
#include <string>

namespace nve_core
{
using NveFeatureVector = std::vector<double>;

enum kEstimators
{
  CTC,
  ACTC,
  HACTC,
  NVEIA
};


/// Discrete states which can be used in the estimation. They are shared between all classifiers
/// In the @binary classification case @inpassable and @passable are used
/// Other states may refer to other types
enum kIAStates
{
  inpassable = 0,
  passable = 1,
  unobserved = 2,
  other = 99
};


/// Adj Modes for Voxels
enum AdjMode_t
{
  NO_ADJ = 0,
  // ADJ8 = 1,
  ADJ26 = 1
};

struct NveRobotFrames_t
{
  std::string world_frame{"map"};
  std::string odom_frame{"odom"};
  std::string robot_frame{"base_link"};
  std::string lidar_frame{"velodyne"};
};

/// @brief Enums for the different types of features we are interrested
enum VoxelFeatures_t
{
  OCCUPANCY = 1,
  MEAN = 2,
  INTENSITY = 3,
  PERMEABILITY = 4,
  SECOND_RETURNS = 5,
  SEMANTIC_LABEL = 6,
  COLOUR = 7,
  EV = 8,
  DECAY = 9,
  NDT = 10
};

}  // namespace nve_core

#endif