// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz

#ifndef COLOUR_FUSER_H
#define COLOUR_FUSER_H

#include <ohm/DefaultLayer.h>
#include <ohm/OccupancyMap.h>
#include <ohm/Voxel.h>
#include <ohm/VoxelAppearance.h>
#include <ohm/VoxelMean.h>

#include <Eigen/Core>
#include <vector>


namespace nve_core
{

// class OhmApperanceFuserBase
// {
// public:
//   explicit OhmApperanceFuserBase(std::shared_ptr<ohm::OccupancyMap> map) { map_ = map; };

//   template <class T>
//   virtual void integrate(const std::vector<Eigen::Vector3d> &points, const std::vector<T> &measurements);

//   virtual T getMeasurment(Eigen::Vector3d &point);

//   virtual T getMeasurment(ohm::&point);


// protected:
//   std::shared_ptr<ohm::OccupancyMap> map_;
// }


/** @brief Endpoint Colour fuser
 *  Uses the colourization on point clouds method to fuse in colourized enpoint measurements recursivly
 *  We use the ohm::Voxelapperance to set the colours
 *  Each colour channel is assumed to be independent. 
 **/
class ColourFuserEP
{
public:
  explicit ColourFuserEP(std::shared_ptr<ohm::OccupancyMap> map);

  void integrate(const std::vector<Eigen::Vector3d> &points, const std::vector<Eigen::Vector3d> &colour);

  void integrate(const Eigen::Vector3d &points, const Eigen::Vector3d &colours);

  bool getColour(const Eigen::Vector3d &point, Eigen::Vector3d &colour);

  bool getColour(const ohm::Key &key, Eigen::Vector3d &colour);

  bool getColourAndPosition(const ohm::Key &key, Eigen::Vector3d &pos, Eigen::Vector3d &colour);

  bool setColour(const ohm::Key &key, const Eigen::Vector3f &colour);

private:
  std::shared_ptr<ohm::OccupancyMap> map_;
  ohm::Voxel<ohm::AppearanceVoxel> colour_layer_;
  ohm::AppearanceVoxel colour_voxel_;
};


/// @brief This class is only there to set/retrieve colour information of a voxel as a wrapper
/// TODO: 
///     - Currently the setter is only valid with RGBA voxel values, where each colour is discibed by a uint_8!
///     - Each colour is hence represented as [0-255]
class OhmColourSetter
{
public:
  explicit OhmColourSetter(std::shared_ptr<ohm::OccupancyMap> map);

  bool getColour(const ohm::Key &key, Eigen::Vector3d &colour);

  bool setColour(const ohm::Key &key, const Eigen::Vector3d &colour);

private:

  // auto setRGBAcolour(const ohm::Key &key, const Eigen::Vector3d &colour)->bool;
  std::shared_ptr<ohm::OccupancyMap> map_;
  ohm::Voxel<ohm::VoxelRGBA> colour_layer_;
  ohm::VoxelRGBA colour_voxel_;
};

}  // namespace nve_core

#endif