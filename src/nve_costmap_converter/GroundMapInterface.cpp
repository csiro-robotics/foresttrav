// MIT
//
// Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO) 
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz

#include "GroundMapInterface.h"

namespace nve_ros
{

auto GroundMapInterface::addGroundKey(ohm::Key const &key) -> void
{
  // Get centroid and get key for x, y, z = 0
  auto key_2d = cellKey(key);
  Cell cell{};
  cell.key = key;
  ground_map_.insert(std::make_pair(key_2d, cell));
}

auto GroundMapInterface::cellKey(ohm::Key const &key) -> ohm::Key
{
  auto pos = map_->voxelCentreGlobal(key);
  return map_->voxelKey(glm::dvec3(pos.x, pos.y, 0.0));
}

auto GroundMapInterface::getGroundKey(ohm::Key const &key, ohm::Key &ground_key) -> bool
{
  auto key_2d = cellKey(key);

  if (ground_map_.find(key_2d) == ground_map_.end())
    return false;

  ground_key = ground_map_.at(key_2d).key;
  return true;
}

auto GroundMapInterface::getCell(ohm::Key const &key, Cell &cell) -> bool
{
  auto key_2d = cellKey(key);

  // Return an emty cell? Should initialised?
  if (ground_map_.find(key_2d) == ground_map_.end())
  {
    return false;
  }

  cell = ground_map_.at(key_2d);
  return true;
}

auto GroundMapInterface::setCell(ohm::Key const &key, Cell const &cell) -> void
{
  auto key_2d = cellKey(key);
  if (not hasKey(key_2d))
  {
    ground_map_.insert(std::make_pair(key_2d, cell));
    return;
  }
  ground_map_.at(key_2d) = cell;
  return;
}

auto GroundMapInterface::setCell(ohm::Key const &key, double z, double cost, bool is_real, bool is_known) -> void
{
  auto key_2d = cellKey(key);
  Cell cell{};
  cell.key = key;
  cell.cost = cost;
  cell.is_real = is_real;
  cell.is_known = is_known;
  cell.z = z;
  this->setCell(key_2d, cell);
  return;
}

auto GroundMapInterface::getCost(ohm::Key const &key) -> double
{
  return ground_map_.at(cellKey(key)).cost;
}

auto GroundMapInterface::setLowestZ(ohm::Key const &key, double z) -> void
{
  auto key_2d = cellKey(key);

  /// If it has a ground key, check that z is larger
  if (hasGroundKey(key_2d))
  {
    /// Lower z exist, do noting
    if (z > ground_map_.at(key_2d).z)
      return;

    /// Lower cell, so we need to update it:
    ground_map_.at(key_2d).z = z;
    ground_map_.at(key_2d).key = key;
  }

  /// New lowest measurement:
  this->setCell(key, z, -1, true, false);

  return;
}

auto GroundMapInterface::getZ(ohm::Key const &key) -> double
{
  auto key_2d = cellKey(key);
  return ground_map_.at(key_2d).z;
}

auto GroundMapInterface::hasKey(ohm::Key const &key) -> bool
{
  return ground_map_.find(key) != ground_map_.end();
}


auto GroundMapInterface::hasGroundKey(ohm::Key const &key) -> bool
{
  return hasKey(cellKey(key));
}


auto GroundMapInterface::allKeys() -> std::vector<ohm::Key>
{
  std::vector<ohm::Key> keys{};
  keys.reserve(ground_map_.size());

  for (auto &key_val : ground_map_)
  {
    keys.emplace_back(key_val.second.key);
  }

  return keys;
}

auto GroundMapInterface::reset() -> void
{
  ground_map_.clear();
}

}  // namespace nve_ros
