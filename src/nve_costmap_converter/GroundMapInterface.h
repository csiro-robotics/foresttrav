// MIT
//
// Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO) 
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz
#ifndef GROUND_MAP_INTERFACE_H
#define GROUND_MAP_INTERFACE_H

#include <ohm/OccupancyMap.h>
#include <glm/vec3.hpp>

#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace nve_ros
{

/// Different ground states {unknown, occupied, free, estimated}

struct Cell
{
  Cell() = default;

  Cell(ohm::Key _key, double _z, double _cost, bool _is_real, bool _is_known)
    : key(_key)
    , cost(_cost)
    , z(_z)
    , is_real(_is_real)
    , is_known(_is_known){};

  ohm::Key key{};
  double cost{ -1.0 };
  double z{ -10.0 };
  bool is_real{ 0 };
  bool is_known{ 0 };
};


class GroundMapInterface
{
public:
  explicit GroundMapInterface(std::shared_ptr<ohm::OccupancyMap> map)
    : map_(map){};

  ///@brief Add a ground key to the ground map
  /// @param key The key to add to the ground map
  /// @return void
  auto addGroundKey(ohm::Key const &key) -> void;

  /// @brief Returns the key (x,y,z) returns the ground key (x,y,z) for ground key(x,y,0)
  /// @param key querry key defined by (x,y,z)_querry
  /// @param ground_key ground key (x,y,z)_{ground} stored at hash_key (x,y,0)
  /// @return true if key found, false otherwise
  auto getGroundKey(ohm::Key const &key, ohm::Key &ground_key) -> bool;

  /// @brief Checks if the ground key exists
  auto hasGroundKey(ohm::Key const &key) -> bool;

  ///@brief Generates the 2d cell key from the general 3D ohm key
  auto cellKey(ohm::Key const &key) -> ohm::Key;

  /// @brief  Reference to a ground cell.
  /// @param key Key to the reference
  /// @return
  auto getCell(ohm::Key const &key, Cell &cell) -> bool;

  /// @brief Sets a ground cell
  /// @param key
  /// @param cell
  auto setCell(ohm::Key const &key, Cell const &cell) -> void;

  /// @brief Set the ground cell with the default values
  auto setCell(ohm::Key const &key, double z, double cost, bool is_real, bool is_known) -> void;

  /// Get the GroundCost of a cell
  auto getCost(ohm::Key const &key) -> double;

  /// @brief Function used to populate the ground map when processing a point cloud
  auto setLowestZ(ohm::Key const &key, double z) -> void;

  auto getZ(ohm::Key const &key) -> double;

  /// @brief checks if we have the ground key for a given voxel set
  auto hasKey(ohm::Key const &key) -> bool;

  /// @brief resets and clears all the data structure
  auto reset() -> void;

  /// @brief returns all keys considered ground map
  auto allKeys() -> std::vector<ohm::Key>;

  /// @brief Reference to the ground map
  auto getGroundMap() -> std::unordered_map<ohm::Key, Cell> const & { return ground_map_; };

private:
  std::shared_ptr<ohm::OccupancyMap> map_;
  std::unordered_map<ohm::Key, Cell> ground_map_;

  mutable std::shared_mutex mutex_;
};


}  // namespace nve_ros


#endif
