#include "GroundMapInterface.h"

namespace nve_core
{

auto GroundMapInterface::addGroundKey(const ohm::Key &key) -> void
{
  // Get centroid and get key for x, y, z = 0
  auto pos = map_->voxelCentreGlobal(key);
  auto key_2d = map_->voxelKey(glm::dvec3(pos.x, pos.y, 0.0));
  ground_map_.insert(std::make_pair(key_2d, key));
}


auto GroundMapInterface::getGroundKey(const ohm::Key &key, ohm::Key &ground_key) -> bool
{
  auto key_2d = map_->voxelKey(map_->voxelCentreGlobal(key));

  if (ground_map_.find(key_2d) == ground_map_.end())
  {
    return false;
  }

  ground_key = ground_map_.at(key_2d);
  return true;
}

auto GroundMapInterface::hasKey(const ohm::Key key) -> bool
{
  return ground_map_.find(key) != ground_map_.end();
}


auto GroundMapInterface::hasGroundKey(const ohm::Key key) -> bool
{
  std::unique_lock lock(mutex_);
  auto pos = map_->voxelCentreGlobal(key);
  auto key_2d = map_->voxelKey(glm::dvec3(pos.x, pos.y, 0.0));
  return hasKey(key_2d);
}


auto GroundMapInterface::allKeys() -> std::vector<ohm::Key>
{
  std::vector<ohm::Key> keys{};
  keys.reserve(ground_map_.size());

  for (auto &key_val : ground_map_)
  {
    keys.emplace_back(key_val.second);
  }

  return keys;
}

auto GroundMapInterface::reset() -> void
{
  ground_map_.clear();
  ground_state_.clear();
}

}  // namespace nve_core
