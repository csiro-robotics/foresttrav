#include "FeatureExtractor.h"
#include <ohm/CovarianceVoxelCompute.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace nve_core
{

size_t FeatureExtractorBase::getFeatureId(const std::string &freature_name)
{
  if (feature_id_map_.empty())
  {
    for (size_t i = 0; i < header_.size(); i++)
    {
      feature_id_map_[header_[i]] = i;
    }
  }

  if (feature_id_map_.find(freature_name) != feature_id_map_.end())
  {
    return feature_id_map_[freature_name];
  }

  assert(false);
  return -1;
}


std::vector<std::vector<double>> FeatureExtractor::extractFeatureByRegion(
  const ohm::KeyRange &key_range, std::unordered_map<ohm::Key, unsigned int> &key_to_featureid)
{
  std::vector<std::vector<double>> feature_maxtrix;
  feature_maxtrix.reserve(1e7);

  // Extract features and get unordered keys
  std::unordered_set<ohm::Key> empty_voxel_keys{};
  for (const auto &key_i : key_range)
  {
    std::vector<double> feature_vi;
    if (!voxelFeature(key_i, feature_vi))
    {
      empty_voxel_keys.insert(key_i);
      continue;
    }
    feature_maxtrix.emplace_back(feature_vi);

    key_to_featureid[key_i] = feature_maxtrix.size() - 1;
  }

  /// Generate the neighboor feature list
  for (auto it = key_to_featureid.begin(); it != key_to_featureid.end(); it++)
  {
    auto local_keys = local_key_range(it->first, empty_voxel_keys);
    std::vector<double> &features_i = feature_maxtrix[it->second];

    /// Generate the mean features
    std::vector<double> mean_features(features_i.size() - 3, 0.0);
    for (const auto &local_key : local_keys)
    {
      // If the voxel is empty continue
      if (empty_voxel_keys.find(local_key) != empty_voxel_keys.end())
      {
        continue;
      }

      for (size_t i = 3; i < features_i.size(); i++)
      {
        mean_features[i - 3] += features_i[i] / static_cast<double>(local_keys.size());
      }
    }
    features_i.insert(features_i.end(), mean_features.begin(), mean_features.end());
  }
  return feature_maxtrix;
}

std::vector<ohm::Key> FeatureExtractor::local_key_range(const ohm::Key &key,
                                                        const std::unordered_set<ohm::Key> &empty_voxels)
{
  /// ToDo: Use KeyRange here
  std::vector<ohm::Key> local_keys{};

  // Get the adjacant keys
  if (AdjMode_t::ADJ26 == adj_mode_)
  {
    local_keys.reserve(header_.size());
    auto pos = map_->voxelCentreGlobal(key);
    auto map_res = map_->resolution();
    auto key_min = map_->voxelKey(pos - glm::dvec3(map_res, map_res, map_res));
    auto key_max = map_->voxelKey(pos + glm::dvec3(map_res, map_res, map_res));
    auto keys = ohm::KeyRange(key_min, key_max, map_->regionVoxelDimensions());

    for (const ohm::Key &key_i : keys)
    {
      if (empty_voxels.find(key_i) != empty_voxels.end())
      {
        continue;
      }

      if (key_i.isNull() || key_i == key)
      {
        continue;
      }

      local_keys.emplace_back(key_i);
    }
  }
  return local_keys;
}


bool FeatureExtractor::extractFeatureByKey(const ohm::Key &key, std::vector<double> &feature_vector)
{
  feature_vector.clear();
  feature_vector.reserve(header_.size());

  // TODO(rue011): Composition for better branching behaviours
  if (!voxelFeature(key, feature_vector))
  {
    return false;
  }

  // Get Keys
  std::vector<ohm::Key> local_keys = local_key_range(key, std::unordered_set<ohm::Key>{});

  auto success = true;
  if (adj_mode_ != AdjMode_t::NO_ADJ)
  {
    success = (appendFeatureByNeighboorhood(key, local_keys, feature_vector));
  }

  assert(feature_vector.size() == header_.size());
  return success;
}


bool FeatureExtractor::voxelFeature(const ohm::Key &key, std::vector<double> &feature_vector)
{
  ohm::Key key_i = key;
  if (!setAndCheckKey(key_i))
  {
    return false;
  }
  feature_vector.clear();
  feature_vector.reserve(header_.size());
  /// Layer accessor

  auto mean_value = mean_layer_.data();
  //
  if (mean_value.count < minimun_number_of_endpoint_observations_ &&
      minimun_number_of_endpoint_observations_ >= 0)  /// @p Magic Number in the mix
  {
    return false;
  }

  auto pos = ohm::positionUnsafe(mean_layer_);
  feature_vector.emplace_back(pos.x);
  feature_vector.emplace_back(pos.y);
  feature_vector.emplace_back(pos.z);

  /// If there is a semantic class
  if (hasFeature(VoxelFeatures_t::SEMANTIC_LABEL))
  {
    auto semantic_values = semantic_layer_.data();
    feature_vector.emplace_back(semantic_values.label);
    feature_vector.emplace_back(std::exp(semantic_values.prob_label) / (1.0 + std::exp(semantic_values.prob_label)));
  }

  feature_vector.emplace_back(mean_value.count);

  if (hasFeature(VoxelFeatures_t::OCCUPANCY))
  {
    const float &occ_value = occ_layer_.data();
    feature_vector.emplace_back(std::exp(occ_value) / (1.0 + std::exp(occ_value)));
    feature_vector.emplace_back(occ_value);
  }

  if (hasFeature(VoxelFeatures_t::INTENSITY))
  {
    auto intensity_values = intensity_layer_.data();
    feature_vector.emplace_back(intensity_values.intensity_mean);  // intensity mean
    feature_vector.emplace_back(intensity_values.intensity_cov);   // intensity covariance
  }

  if (hasFeature(VoxelFeatures_t::PERMEABILITY))
  {
    auto hm_value = hm_layer_.data();
    feature_vector.emplace_back(hm_value.miss_count);
    feature_vector.emplace_back(hm_value.hit_count);
    double permeability = double(hm_value.hit_count) + double(hm_value.miss_count) > 0.0 ?
                            double(hm_value.miss_count) / (double(hm_value.miss_count) + double(hm_value.hit_count)) :
                            0;
    feature_vector.emplace_back(permeability);
  }

  if (hasFeature(VoxelFeatures_t::NDT))
  {
      /// Trianglar square root covariance matrix. Represents a covariance matrix via the triangular
      /// square root matrix, P = S * S^T.
      /// | cov[0]  |      .  |      .  |
      /// | cov[1]  | cov[2]  |      .  |
      /// | cov[3]  | cov[4]  | cov[5]  |
    cov_layer_.read(&cov_value_);
    feature_vector.emplace_back(cov_value_.trianglar_covariance[0]);
    feature_vector.emplace_back(cov_value_.trianglar_covariance[1]);
    feature_vector.emplace_back(cov_value_.trianglar_covariance[3]);
    feature_vector.emplace_back(cov_value_.trianglar_covariance[2]);
    feature_vector.emplace_back(cov_value_.trianglar_covariance[4]);
    feature_vector.emplace_back(cov_value_.trianglar_covariance[5]);
  }

  if (hasFeature(VoxelFeatures_t::DECAY))
  {
    auto traversal_data = trav_layer_.data();
    feature_vector.emplace_back(traversal_data);
  }

  if (hasFeature(VoxelFeatures_t::SECOND_RETURNS))
  {
    auto second_return_values = second_return_layer_.data();
    feature_vector.emplace_back(second_return_values.count);
    feature_vector.emplace_back(ohm::secondarySampleRangeMean(second_return_values));
    feature_vector.emplace_back(ohm::secondarySampleRangeStdDev(second_return_values));  // 18#include <algorithm>
  }

  if (hasFeature(VoxelFeatures_t::COLOUR))
  {
    auto colour = appearance_layer_.data();
    feature_vector.emplace_back(colour.red[0]);
    feature_vector.emplace_back(colour.green[0]);
    feature_vector.emplace_back(colour.blue[0]);
    feature_vector.emplace_back(colour.count);
  }
  /// This should blow up if the feature vector is not the same size as the header
  if (hasFeature(VoxelFeatures_t::EV))
  {
    cov_layer_.read(&cov_value_);
    // Eigenvalue Features
    // Eigen Values are order in ascending(Checked) oder, always!
    Eigen::Matrix3d covar;
    covFromGlmToEigen(cov_value_.trianglar_covariance, covar);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
    es.compute(covar);

    Eigen::Vector3d ev_0 = es.eigenvectors().col(0);
    double theta = gravityAlignedAngle(ev_0);

    /// Eigenvalues
    /// EV_0 <= EV_1 <= EV_2
    feature_vector.emplace_back(es.eigenvalues()[0]);                                                // ndt-roughness
    feature_vector.emplace_back(theta);                                          // angle
    // Lalonde, Jean‐François, et al. "Natural terrain classification using three‐dimensional ladar
    feature_vector.emplace_back((es.eigenvalues()[2] - es.eigenvalues()[1]) / es.eigenvalues()[2]);  // linear
    feature_vector.emplace_back((es.eigenvalues()[1] - es.eigenvalues()[0]) / es.eigenvalues()[2]);  // planar
    feature_vector.emplace_back(es.eigenvalues()[0] / es.eigenvalues()[2]);                          // spherical 15
  }


  for (int i = 0; i < feature_vector.size(); i++)
  {
    if (std::isnan(feature_vector[i]) || std::isinf(feature_vector[i]))
      feature_vector[i] = 0.0;
  }

  return true;
}

void covFromGlmToEigen(float glm_cov[6], Eigen::Matrix3d &cov_i)
{
  Eigen::Matrix3d covar_temp = Eigen::Matrix3d::Zero();
  covar_temp(0, 0) = glm_cov[0];
  covar_temp(1, 0) = glm_cov[1];
  covar_temp(1, 1) = glm_cov[3];
  covar_temp(2, 0) = glm_cov[2];
  covar_temp(2, 1) = glm_cov[4];
  covar_temp(2, 2) = glm_cov[5];
  cov_i = covar_temp * covar_temp.transpose();
}

double gravityAlignedAngle(Eigen::Vector3d &ev)
{
  Eigen::Vector3d eg(0, 0, 1);
  double theta = std::abs(std::acos(eg.dot(ev)));

  // Smarter way to do this?
  if (theta > M_PI_2)
  {
    theta = std::abs(std::acos(eg.dot(-1.0 * ev)));
  }
  return theta;
}

bool FeatureExtractor::hasFeature(VoxelFeatures_t feature)
{
  return (feature_set_ & (1 << uint32_t(feature))) != 0;
}

bool FeatureExtractor::checkSetup()
{
  /// Check if the map is valid
  const ohm::MapLayout &layout = map_->layout();
  header_.clear();
  feature_id_map_.clear();

  // Mean
  feature_set_ |= 1 << uint32_t(VoxelFeatures_t::MEAN);

  header_.emplace_back("x");
  header_.emplace_back("y");
  header_.emplace_back("z");


  if (layout.semanticLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::SEMANTIC_LABEL);
    header_.emplace_back("label");
    header_.emplace_back("label_prob");
  }

  header_.emplace_back("mean_count");
  if (layout.occupancyLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::OCCUPANCY);
    header_.emplace_back("occupancy_prob");
    header_.emplace_back("occupancy_log_probability");
  }

  if (layout.intensityLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::INTENSITY);
    header_.emplace_back("intensity_mean");
    header_.emplace_back("intensity_covariance");
  }

  if (layout.hitMissCountLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::PERMEABILITY);
    header_.emplace_back("miss_count");
    header_.emplace_back("hit_count");
    header_.emplace_back("permeability");
  }

  if (layout.covarianceLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::NDT);
    header_.emplace_back("covariance_xx_sqrt");
    header_.emplace_back("covariance_xy_sqrt");
    header_.emplace_back("covariance_xz_sqrt");
    header_.emplace_back("covariance_yy_sqrt");
    header_.emplace_back("covariance_yz_sqrt");
    header_.emplace_back("covariance_zz_sqrt");
  }

  if (layout.traversalLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::DECAY);
    header_.emplace_back("traversal");
  }

  if (layout.secondarySamplesLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::SECOND_RETURNS);
    header_.emplace_back("secondary_sample_count");
    header_.emplace_back("secondary_sample_range_mean");
    header_.emplace_back("secondary_sample_range_std_dev");
  }

  if (layout.appearanceLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::COLOUR);
    header_.emplace_back("red");
    header_.emplace_back("green");
    header_.emplace_back("blue");
    header_.emplace_back("colour_count");
  }

  if (layout.covarianceLayer() >= 0)
  {
    feature_set_ |= 1 << uint32_t(VoxelFeatures_t::EV);
    header_.emplace_back("ndt_rho");
    header_.emplace_back("theta");
    header_.emplace_back("ev_lin");
    header_.emplace_back("ev_plan");
    header_.emplace_back("ev_sph");  // 15
  }

  /// Set adjacency header
  /// TODO: Remove mean_pos....
  if (adj_mode_ != AdjMode_t::NO_ADJ)
  {
    std::vector<std::string> adj_header;
    adj_header.reserve(header_.size());

    for (std::string feature_topic : header_)
    {
      if ("x" == feature_topic)
        continue;

      if ("y" == feature_topic)
        continue;

      if ("z" == feature_topic)
        continue;

      if ("label" == feature_topic)
        continue;

      if ("label_prob" == feature_topic)
        continue;

      adj_header.push_back("adj_" + feature_topic);
    }
    header_.insert(header_.end(), adj_header.begin(), adj_header.end());
  }
  return true;
}

/// TODO: This is incredible inefficient but ok for now.....
bool FeatureExtractor::appendFeatureByNeighboorhood(const ohm::Key &original_key, const std::vector<ohm::Key> &keys,
                                                    std::vector<double> &features)
{ 
  // This is fragile magic indexing at its worts. 
  // The offset 5 is from [x,y,z,label,label_prob] so we start with mean_count at 5
  // If it is 3 it can be [x,y,z] when there are no semantic labels..
  int magic_index_spacing = 3;
  if (hasFeature(VoxelFeatures_t::SEMANTIC_LABEL))
  {
    magic_index_spacing = 5;
  }

  double num_valid_voxels{ 0.0 };

  /// Generate the mean features
  std::vector<double> mean_features(features.size() - magic_index_spacing, 0.0);
  for (ohm::Key key_i : keys)
  {
    /// Skip the original key
    if (key_i == original_key)
    {
      continue;
    }

    /// Get feature values for adjacent voxel
    std::vector<double> adj_voxel_features(features.size() - magic_index_spacing, 0);
    if (!voxelFeature(key_i, adj_voxel_features))
    {
      continue;
    }

    // Add the value to the mean_feature vector
    for (size_t i = magic_index_spacing; i < adj_voxel_features.size(); i++)
    {
      mean_features[i - magic_index_spacing] += adj_voxel_features[i];
    }
    num_valid_voxels++;
  }

  if (num_valid_voxels > 0)
  /// Normalize the values based on the number of valid voxels
  {
    for (size_t i = magic_index_spacing; i < features.size(); i++)
    {
      mean_features[i - magic_index_spacing] /= num_valid_voxels;
    }
  }

  features.insert(features.end(), mean_features.begin(), mean_features.end());

  return true;
}


bool FeatureExtractor::setAndCheckKey(ohm::Key &key)
{
  ohm::setVoxelKey(key, mean_layer_, occ_layer_, intensity_layer_, hm_layer_, cov_layer_, second_return_layer_,
                   semantic_layer_, trav_layer_, appearance_layer_);

  bool valid = true;
  /// Semantic Label is always first
  if (!(semantic_layer_.isValid()) && hasFeature(VoxelFeatures_t::SEMANTIC_LABEL))
  {
    std::cout << "Semantic layer not valid" << std::endl;
    valid = false;
  }
  if (!(mean_layer_.isValid()))
  {
    std::cout << "Mean layer not valid" << std::endl;
    valid = false;
  }
  if (!(occ_layer_.isValid()) && hasFeature(VoxelFeatures_t::OCCUPANCY))
  {
    std::cout << "Occupancy layer not valid" << std::endl;
    valid = false;
  }
  if (!(intensity_layer_.isValid()) && hasFeature(VoxelFeatures_t::INTENSITY))
  {
    std::cout << "Intensity layer not valid" << std::endl;
    valid = false;
  }
  if (!(hm_layer_.isValid()) && hasFeature(VoxelFeatures_t::PERMEABILITY))
  {
    std::cout << "hm layer not valid" << std::endl;
    valid = false;
  }
  if (!(cov_layer_.isValid()) && hasFeature(VoxelFeatures_t::EV))
  {
    std::cout << "Cov layer not valid" << std::endl;
    valid = false;
  }
  if (!(second_return_layer_.isValid()) && hasFeature(VoxelFeatures_t::SECOND_RETURNS))
  {
    std::cout << "Second Return layer not valid" << std::endl;
    valid = false;
  }
  if (!(trav_layer_.isValid()) && hasFeature(VoxelFeatures_t::DECAY))
  {
    std::cout << "Traversal layer not valid" << std::endl;
    valid = false;
  }
  if (!(appearance_layer_.isValid()) && hasFeature(VoxelFeatures_t::COLOUR))
  {
    std::cout << "apperance layer not valid" << std::endl;
    valid = false;
  }

  return valid;
}


}  // namespace nve_core
