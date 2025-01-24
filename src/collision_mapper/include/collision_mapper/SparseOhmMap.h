/* MIT License
 * Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO) 
 * Queensland University of Technology (QUT)
 *
 * Author: Fabio Ruetz
 */
#ifndef SPARSE_OHM_MAP_H
#define SPARSE_OHM_MAP_H

#include <ohm/ohm/OccupancyMap.h>
#include <glm/vec3.hpp>
#include <iterator>
#include <memory>
#include <unordered_map>

namespace sohm
{

template <typename T>
class SparseOhmMap
{
public:
  SparseOhmMap(float voxel_size);

  auto write(const glm::dvec3 &pos, T value) -> void;
  auto write(const ohm::Key &key, T value) -> void;
  auto exists(const ohm::Key &key) -> bool;
  auto exists(const ohm::Key &key) const -> bool;
  auto read(const ohm::Key &key, T &value) -> bool;
  auto read(const ohm::Key &key, T &value) const -> bool;
  auto clear() -> void;

  // Iterator class definition
  class iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<const ohm::Key, T>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = value_type &;

    iterator(typename std::unordered_map<ohm::Key, T>::iterator it)
      : it_(it)
    {}

    reference operator*() const { return *it_; }
    pointer operator->() const { return &(*it_); }

    iterator &operator++()
    {
      ++it_;
      return *this;
    }
    iterator operator++(int)
    {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const iterator &a, const iterator &b) { return a.it_ == b.it_; }
    friend bool operator!=(const iterator &a, const iterator &b) { return a.it_ != b.it_; }

  private:
    typename std::unordered_map<ohm::Key, T>::iterator it_;
  };

  // Const iterator class definition
  class const_iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const std::pair<const ohm::Key, T>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    const_iterator(typename std::unordered_map<ohm::Key, T>::const_iterator it)
      : it_(it)
    {}

    reference operator*() const { return *it_; }
    pointer operator->() const { return &(*it_); }

    const_iterator &operator++()
    {
      ++it_;
      return *this;
    }
    const_iterator operator++(int)
    {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const const_iterator &a, const const_iterator &b) { return a.it_ == b.it_; }
    friend bool operator!=(const const_iterator &a, const const_iterator &b) { return a.it_ != b.it_; }

  private:
    typename std::unordered_map<ohm::Key, T>::const_iterator it_;
  };

  // Begin and end methods for iterator
  iterator begin() { return iterator(sparse_map_.begin()); }
  iterator end() { return iterator(sparse_map_.end()); }

  const_iterator begin() const { return const_iterator(sparse_map_.begin()); }
  const_iterator end() const { return const_iterator(sparse_map_.end()); }

  const_iterator cbegin() const { return const_iterator(sparse_map_.cbegin()); }
  const_iterator cend() const { return const_iterator(sparse_map_.cend()); }

private:
  std::unique_ptr<ohm::OccupancyMap> map_;
  std::unordered_map<ohm::Key, T> sparse_map_;
};

template <typename T>
SparseOhmMap<T>::SparseOhmMap(float voxel_size)
{
  map_ = std::make_unique<ohm::OccupancyMap>(voxel_size);
}

template <typename T>
auto SparseOhmMap<T>::write(const glm::dvec3 &pos, T value) -> void
{
  const ohm::Key key = map_->voxelKey(pos);
  write(key, value);
}

template <typename T>
auto SparseOhmMap<T>::write(const ohm::Key &key, T value) -> void
{
  if (exists(key))
  {
    sparse_map_.at(key) = value;
    return;
  }

  sparse_map_.insert(std::make_pair(key, value));
}

template <typename T>
auto SparseOhmMap<T>::exists(const ohm::Key &key) -> bool
{
  return !(sparse_map_.find(key) == sparse_map_.end());
}

template <typename T>
auto SparseOhmMap<T>::exists(const ohm::Key &key) const -> bool
{
  return !(sparse_map_.find(key) == sparse_map_.end());
}

template <typename T>
auto SparseOhmMap<T>::read(const ohm::Key &key, T &value) -> bool
{
  if (exists(key))
  {
    value = sparse_map_.at(key);
    return true;
  }

  return false;
}

template <typename T>
auto SparseOhmMap<T>::read(const ohm::Key &key, T &value) const -> bool
{
  if (exists(key))
  {
    value = sparse_map_.at(key);
    return true;
  }

  return false;
}

template <typename T>
auto SparseOhmMap<T>::clear() -> void
{
  sparse_map_.clear();
}

}  // namespace sohm

#endif  // SPARSE_OHM_MAP_H
