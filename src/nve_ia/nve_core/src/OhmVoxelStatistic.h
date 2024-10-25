// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz
#ifndef OHM_VOXEL_STATISTICS
#define OHM_VOXEL_STATISTICS

#include "FeatureExtractor.h"

// #include "ColourFuser.h"

// #include <Eigen/Core>

// #include <ohm/DefaultLayer.h>
// #include <ohm/Key.h>
// #include <ohm/KeyList.h>
// #include <ohm/KeyRange.h>
// #include <ohm/MapChunk.h>
// #include <ohm/MapLayer.h>
// #include <ohm/MapLayout.h>
// #include <ohm/MapSerialise.h>
// #include <ohm/NdtMap.h>
// #include <ohm/OccupancyMap.h>
// #include <ohm/Trace.h>
// #include <ohm/Voxel.h>
// #include <ohm/VoxelAppearance.h>
// #include <ohm/VoxelData.h>
// #include <ohm/VoxelSemanticLabel.h>


namespace nve_core
{

class OhmVoxelStatistic:public FeatureExtractorBase
{
public:
  explicit OhmVoxelStatistic(std::shared_ptr<ohm::OccupancyMap> map, bool use_adjacency_features = true)
    : map_(map){
      auto b = checkSetup();
      use_adjacency_features_ = use_adjacency_features;
    };
  ~OhmVoxelStatistic() = default;

  /// @brief Extracts the voxel statistics based on an ohm Key
  /// @param key
  /// @param feature_vector
  /// @return True if the voxel has sufficient information, false otherwise
  bool extractFeatureByKey(const ohm::Key &key, std::vector<double> &feature_vector);

  bool extractVolxeFeature(const ohm::Key &key, std::vector<double> &feature_vector);


  bool checkSetup() override;

  /// @brief
  /// @return
  ohm::OccupancyMap &occupancyMap() const { return *map_; }

  /// @brief Checks ifa  feature is peresent
  /// @param feature Defined by VoxelFeatures_t
  /// @return True if the feature exists, false otherwise
  bool hasFeature(VoxelFeatures_t feature) { return (feature_set_ & (1 << uint32_t(feature))) != 0; };

  /// @brief Sets the feture set we want to compute
  /// @param target_feature_set 
  void setFeatureSet(uint32_t target_feature_set)
  {
    feature_set_ = target_feature_set;
  }
  

private:
  std::shared_ptr<ohm::OccupancyMap> map_;
  int minimun_number_of_endpoint_observations_{};
  bool use_adjacency_features_{};
};


}  // namespace nve_core

#endif
