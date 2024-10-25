#include "OhmVoxelStatistic.h"
#include <ohm/VoxelSecondarySample.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>

namespace nve_core
{


bool OhmVoxelStatistic::extractFeatureByKey(const ohm::Key &key, std::vector<double> &feature_vector)
{ 
  feature_vector.clear();
  feature_vector.reserve(header_.size());
  /// Layer accessor

  ohm::Voxel<const ohm::VoxelMean> mean_layer_(&occupancyMap(), map_->layout().meanLayer());
  mean_layer_.setKey(key);
  auto mean_value = mean_layer_.data();
  //
  if (mean_value.count < minimun_number_of_endpoint_observations_ &&
      minimun_number_of_endpoint_observations_ >= 0)  /// @p Magic Number in the mix
  {
    return false;
  }

  auto pos = ohm::positionSafe(mean_layer_);
  feature_vector.emplace_back(pos.x);
  feature_vector.emplace_back(pos.y);
  feature_vector.emplace_back(pos.z);
  


  /// If there is a semantic class
  if (hasFeature(VoxelFeatures_t::SEMANTIC_LABEL))
  {
    ohm::Voxel<const ohm::SemanticLabel> semantic_layer(&occupancyMap(), map_->layout().semanticLayer());
    semantic_layer.setKey(key);
    auto semantic_values = semantic_layer.data();
    feature_vector.emplace_back(semantic_values.label);
    feature_vector.emplace_back(std::exp(semantic_values.prob_label) / (1 + std::exp(semantic_values.prob_label)));
  }

  // This has to live after the label and label_prob
  feature_vector.emplace_back(mean_value.count);

  if (hasFeature(VoxelFeatures_t::OCCUPANCY))
  {
    ohm::Voxel<const float> occ_layer_(&occupancyMap(), map_->layout().occupancyLayer());
    occ_layer_.setKey(key);
    const float& occ_value = occ_layer_.data();
    feature_vector.emplace_back(std::exp(occ_value) / (1 + std::exp(occ_value)));
  }

  if (hasFeature(VoxelFeatures_t::INTENSITY))
  {
    ohm::Voxel<const ohm::IntensityMeanCov> voxel_intensity(&occupancyMap(), map_->layout().intensityLayer());
    voxel_intensity.setKey(key);
    auto intensity_values = voxel_intensity.data();
    feature_vector.emplace_back(intensity_values.intensity_mean);  // intensity mean
    feature_vector.emplace_back(intensity_values.intensity_cov);   // intensity covariance
  }

  if (hasFeature(VoxelFeatures_t::PERMEABILITY))
  {
    ohm::Voxel<const ohm::HitMissCount> hm_layer_(&occupancyMap(), map_->layout().hitMissCountLayer());
    hm_layer_.setKey(key);
    auto hm_value = hm_layer_.data();
    feature_vector.emplace_back(hm_value.miss_count);
    feature_vector.emplace_back(hm_value.hit_count);
    double permeability = double(hm_value.hit_count) + double(hm_value.miss_count) > 0.0 ? double(hm_value.miss_count)  / ( double(hm_value.miss_count) +  double(hm_value.hit_count)) : 0;
    feature_vector.emplace_back(permeability);
  }

  if (hasFeature(VoxelFeatures_t::NDT))
  {
    ohm::Voxel<const ohm::CovarianceVoxel> voxel_covariance(&occupancyMap(), map_->layout().covarianceLayer());
    voxel_covariance.setKey(key);
    auto glm_cov = voxel_covariance.data().trianglar_covariance;
    feature_vector.emplace_back(glm_cov[0]);
    feature_vector.emplace_back(glm_cov[1]);
    feature_vector.emplace_back(glm_cov[2]);
    feature_vector.emplace_back(glm_cov[3]);
    feature_vector.emplace_back(glm_cov[4]);
    feature_vector.emplace_back(glm_cov[5]);
  }

  if (hasFeature(VoxelFeatures_t::DECAY))
  {
    ohm::Voxel<const float> voxel_traversal(&occupancyMap(), map_->layout().traversalLayer());
    voxel_traversal.setKey(key);
    float trav_length = voxel_traversal.data();
    feature_vector.emplace_back(voxel_traversal.data());
  }

  if (hasFeature(VoxelFeatures_t::SECOND_RETURNS))
  {  
    ohm::Voxel<ohm::VoxelSecondarySample> second_return_layer(map_.get(), map_->layout().secondarySamplesLayer());
    // second_return_layer.setKey(key);
    ohm::setVoxelKey(key, second_return_layer); 
    auto second_return_values = second_return_layer.data();
    feature_vector.emplace_back(second_return_values.count);
    feature_vector.emplace_back(ohm::secondarySampleRangeMean(second_return_values));
    feature_vector.emplace_back(ohm::secondarySampleRangeStdDev(second_return_values));  // 18
  }

  if (hasFeature(VoxelFeatures_t::COLOUR))
  {
    ohm::Voxel<const ohm::AppearanceVoxel> voxel_appearance(&occupancyMap(), map_->layout().appearanceLayer());
    voxel_appearance.setKey(key);
    auto colour = voxel_appearance.data();
    // colour_setter_->getColour(key, colour);
    feature_vector.emplace_back(colour.red[0]);
    feature_vector.emplace_back(colour.green[0]);
    feature_vector.emplace_back(colour.blue[0]);
    feature_vector.emplace_back(colour.count);
  }
  /// This should blow up if the feature vector is not the same size as the header
   if (hasFeature(VoxelFeatures_t::EV))
  {
    ohm::Voxel<const ohm::CovarianceVoxel> voxel_covariance(&occupancyMap(), map_->layout().covarianceLayer());
    voxel_covariance.setKey(key);
    auto glm_cov = voxel_covariance.data().trianglar_covariance;

    // Eigenvalue Features
    // Eigen Values are order in ascending(Checked) oder, always!
    Eigen::Matrix3d covar;
    covFromGlmToEigen(glm_cov, covar);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
    es.compute(covar);

    Eigen::Vector3d ev_0 = es.eigenvectors().col(0);
    double theta = gravityAlignedAngle(ev_0);

    /// Eigenvalues
    /// EV_0 <= EV_1 <= EV_2
    feature_vector.emplace_back(es.eigenvalues()[0]);                                                // ndt-roughness
    feature_vector.emplace_back(theta);
    /// Based on lalonde                                                              // angle
    feature_vector.emplace_back((es.eigenvalues()[2] - es.eigenvalues()[1]) / es.eigenvalues()[2]);  // linear
    feature_vector.emplace_back((es.eigenvalues()[1] - es.eigenvalues()[0]) / es.eigenvalues()[2]);  // planar
    feature_vector.emplace_back(es.eigenvalues()[0] / es.eigenvalues()[2]);                          // spherical 15
  }


  assert(feature_vector.size() == header_.size());
  return true;
}

bool OhmVoxelStatistic::checkSetup()
{
  /// Check if the map is valid
  const ohm::MapLayout& layout = map_->layout();
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
    header_.emplace_back("cov_xx");
    header_.emplace_back("cov_xy");
    header_.emplace_back("cov_xz");
    header_.emplace_back("cov_yy");
    header_.emplace_back("cov_yz");
    header_.emplace_back("cov_zz");
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

}

}  // namespace nve_core
