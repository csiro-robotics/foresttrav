// Copyright (c) 2022
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz
#ifndef FEATURES_EXTRACTOR_H
#define FEATURES_EXTRACTOR_H

// #include "NdtFeatures.h"
#include "Types.h"

#include "ColourFuser.h"

#include <Eigen/Core>

#include <ohm/DefaultLayer.h>
#include <ohm/Key.h>
#include <ohm/KeyList.h>
#include <ohm/KeyRange.h>
#include <ohm/MapChunk.h>
#include <ohm/MapLayer.h>
#include <ohm/MapLayout.h>
#include <ohm/MapSerialise.h>
#include <ohm/NdtMap.h>
#include <ohm/OccupancyMap.h>
#include <ohm/Trace.h>
#include <ohm/Voxel.h>
#include <ohm/VoxelAppearance.h>
#include <ohm/VoxelData.h>
#include <ohm/VoxelSemanticLabel.h>

#include <map>
#include <unordered_set>

namespace nve_core
{

void covFromGlmToEigen(float glm_cov[6], Eigen::Matrix3d &cov_i);

double gravityAlignedAngle(Eigen::Vector3d &ev);

/// TODO: Two major problems:
/// 1) Feature name by index feature_id["occ_count"] and not specific ordering to avoid misslabelled header and values
/// 2) Adjacency features no need to recompute all the values?
/// - Compute all the inital values and generate key to index lookup
///   - Do adjacency

class FeatureExtractorBase
{
public:
  virtual ~FeatureExtractorBase() = default;

  virtual bool extractFeatureByKey(const ohm::Key &key, std::vector<double> &feature_vector) = 0;

  /// @brief Check that the feature_set is valid, that the map layers are valid and generates the header and set
  virtual bool checkSetup() = 0;

  /// @brief Accessor to the header
  std::vector<std::string> getFeatureHeader() { return header_; };

  void setFeatureSet(uint32_t target_feature_set) { feature_set_ = target_feature_set; };

  size_t getFeatureId(const std::string &feature_name);

protected:
  uint32_t feature_set_{};                       /// Toggle to enable features
  std::vector<std::string> header_{};            /// Header for the feature set
  std::map<std::string, int> feature_id_map_{};  /// Map to access the feature id by name
};


class FeatureExtractor : public FeatureExtractorBase
{
public:
  // FeatureExtractor() = default;
  explicit FeatureExtractor(std::shared_ptr<ohm::OccupancyMap> map)
    : map_(map)
  {
    occ_layer_ = ohm::Voxel<float>(map_.get(), map_->layout().occupancyLayer());
    mean_layer_ = ohm::Voxel<ohm::VoxelMean>(map_.get(), map_->layout().meanLayer());
    cov_layer_ = ohm::Voxel<ohm::CovarianceVoxel>(map_.get(), map_->layout().covarianceLayer());
    intensity_layer_ = ohm::Voxel<ohm::IntensityMeanCov>(map_.get(), map_->layout().intensityLayer());
    hm_layer_ = ohm::Voxel<ohm::HitMissCount>(map_.get(), map_->layout().hitMissCountLayer());
    second_return_layer_ = ohm::Voxel<ohm::VoxelSecondarySample>(map_.get(), map_->layout().secondarySamplesLayer());
    trav_layer_ = ohm::Voxel<float>(map_.get(), map_->layout().traversalLayer());
    semantic_layer_ = ohm::Voxel<ohm::SemanticLabel>(map_.get(), map_->layout().semanticLayer());
    appearance_layer_ = ohm::Voxel<ohm::AppearanceVoxel>(map_.get(), map_->layout().appearanceLayer());

    adj_mode_ = AdjMode_t::ADJ26;
    checkSetup();

  }
  ~FeatureExtractor() = default;


  // Function which checks what features are avaialble
  bool checkSetup() override;

  /// @brief Extract features by Key or
  bool extractFeatureByKey(const ohm::Key &key, std::vector<double> &feature_vector);

  /// Extract a features for a key_range
  std::vector<std::vector<double>> extractFeatureByRegion(const ohm::KeyRange &key_range,
                                                          std::unordered_map<ohm::Key, unsigned int> &key_to_featureid);

  /// @brief Given a set of keys defining a neighbourhood this will add all features as a mean
  /// @param original_key The key of the original voxel which is used to extract the features
  /// @param keys The keys of the neighbourhood
  /// @param features The feature vector which will be appended the adjance features to
  bool appendFeatureByNeighboorhood(const ohm::Key &original_key, const std::vector<ohm::Key> &keys, std::vector<double> &features);

  /// Returns true if the value is in the feature set
  bool hasFeature(VoxelFeatures_t value);

  /// @brief Returns the header which will be writtend
  std::vector<std::string> getHeader() { return header_; };

  std::vector<std::string> getCCHeader()
  {
    auto cc_header = header_;

    for (auto &file_name : cc_header)
    {
      if (file_name.find("pos") != std::string::npos)
        continue;

      file_name = "scalar_" + file_name;
    }
    return cc_header;
  };

  /// @brief Setter for the adjacency mode
  void setAdj(AdjMode_t mode)
  {
    adj_mode_ = mode;
    checkSetup();
  };


private:
  /// @brief Extracts the features based per voxel
  /// This method extract the voxel features based on a @p ohm::Key and returns a @p feature_vector
  /// It set and checks the keys based on the @c feature_set_, where this class defined what features
  /// it expect to find.
  bool voxelFeature(const ohm::Key &key, std::vector<double> &features_vector);

  /// @brief Sets and check the *_layers of the @c FeatureSelector can be assigned with a @p
  /// @p ohm::Key
  /// Note: Will check the layers based on @c feature_set_ and what has been defined.
  bool setAndCheckKey(ohm::Key &keys);


  //// @brief Finds the the adjacency meassures based on the local key range
  std::vector<ohm::Key> local_key_range(const ohm::Key &key, const std::unordered_set<ohm::Key> &empty_voxels);


  // All layers of ohm
  ohm::Voxel<float> occ_layer_;
  ohm::Voxel<ohm::VoxelMean> mean_layer_;
  ohm::Voxel<ohm::CovarianceVoxel> cov_layer_;
  ohm::Voxel<ohm::IntensityMeanCov> intensity_layer_;
  ohm::Voxel<ohm::HitMissCount> hm_layer_;
  ohm::Voxel<ohm::SemanticLabel> semantic_layer_;
  ohm::Voxel<ohm::VoxelSecondarySample> second_return_layer_;
  ohm::Voxel<ohm::AppearanceVoxel> appearance_layer_;
  ohm::Voxel<float> trav_layer_;

  // Values for ohm to read
  ohm::VoxelMean mean_value_;
  ohm::CovarianceVoxel cov_value_;
  ohm::IntensityMeanCov intensity_value_;
  ohm::HitMissCount hm_value_;
  ohm::SemanticLabel semantic_values_;
  ohm::VoxelSecondarySample second_return_values_;
  ohm::AppearanceVoxel appearance_values_;

  //

  // std::vector<std::string> header_;
  std::shared_ptr<ohm::OccupancyMap> map_;
  AdjMode_t adj_mode_;

  int minimun_number_of_endpoint_observations_{3};
};
}  // namespace nve_core

#endif
