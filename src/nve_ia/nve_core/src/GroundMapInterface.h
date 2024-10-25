// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz
#ifndef GROUND_MAP_INTERFACE_H
#define GROUND_MAP_INTERFACE_H

#include <ohm/OccupancyMap.h>
#include <glm/vec3.hpp>

#include <unordered_map>
#include <mutex>
#include <shared_mutex>

namespace nve_core
{

/// Different ground states {unknown, occupied, free, estimated}
///
enum GroundState
{
  UNKNOWN = 0,
  OCCUPIED = 1,
  FREE = 2,
  ESTIMATED = 3
};


class GroundMapInterface
{
public:
  explicit GroundMapInterface(std::shared_ptr<ohm::OccupancyMap> map)
    : map_(map){};

   ///@brief Add a ground key to the ground map
  /// @param key The key to add to the ground map
  /// @return void
  auto addGroundKey(const ohm::Key &key) -> void;

  /// @brief Add a ground key and state to the ground map
  auto addGroundState(const ohm::Key &key, GroundState state) -> void;

  auto addGroundKeyAndState(const ohm::Key &key, const GroundState state) -> void;

  /// @brief Returns the key (x,y,z) returns the ground key (x,y,z) for ground key(x,y,0)  
  /// @param key querry key defined by (x,y,z)_querry
  /// @param ground_key ground key (x,y,z)_{ground} stored at hash_key (x,y,0)
  /// @return true if key found, false otherwise
  auto getGroundKey(const ohm::Key &key, ohm::Key &ground_key) -> bool;

  /// @brief checks if we have the ground key for a given voxel set
  auto hasKey(const ohm::Key key)-> bool;
  
  auto hasGroundKey(const ohm::Key key)-> bool;

  /// @brief resets and clears all the data structure
  auto reset() -> void;

  /// @brief returns all keys considered ground map
  auto allKeys()-> std::vector<ohm::Key>;

private:
  std::shared_ptr<ohm::OccupancyMap> map_;
  std::unordered_map<ohm::Key, ohm::Key> ground_map_; 
  std::unordered_map<ohm::Key, GroundState> ground_state_;

  mutable std::shared_mutex mutex_;
};


}  // namespace nve_core


#endif