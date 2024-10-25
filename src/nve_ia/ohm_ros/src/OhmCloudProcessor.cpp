#include "OhmCloudProcessor.h"

#include <sensor_msgs/point_cloud2_iterator.h>

#include <ohm/CovarianceVoxel.h>
#include <ohm/DefaultLayer.h>
#include <ohm/Key.h>
#include <ohm/Voxel.h>
#include <ohm/VoxelMean.h>
#include <ohm/VoxelSecondarySample.h>
#include <ohm/VoxelAppearance.h>

namespace ohm_ros
{

void OhmCloudProcessor::processOhmCloud(const sensor_msgs::PointCloud2::ConstPtr msg, Eigen::Affine3d &T_m_message)
{
  /// Mean layer
  BaseCloudIterator pos_iter(msg);
  OccCloudIterator occ_iter(msg);

  /// Check if unique pointers are slow!
  std::unique_ptr<TraversalCloudIterator> trav_iter_ptr{};
  if (params_.traversal_enable)
  {
    trav_iter_ptr = std::make_unique<TraversalCloudIterator>(msg);
  }
  std::unique_ptr<NdtCloudIterator> ndt_iter_ptr{};
  if (params_.ndt_tm_enable)
  {
    ndt_iter_ptr = std::make_unique<NdtCloudIterator>(msg);
  }
  std::unique_ptr<SecondarySampleIterator> sr_iter_ptr{};
  if (params_.second_return_enable)
  {
    sr_iter_ptr = std::make_unique<SecondarySampleIterator>(msg);
  }

  std::unique_ptr<ohm_ros::RGBSampleIterator> rgb_iter_ptr{};
  if (params_.colour_enable)
  {
    rgb_iter_ptr = std::make_unique<ohm_ros::RGBSampleIterator>(msg);
  }


  /// Layers and Voxels for values
  ohm::Voxel<float> occ_layer_(map_.get(), map_->layout().occupancyLayer());
  ohm::Voxel<ohm::VoxelMean> mean_layer_(map_.get(), map_->layout().meanLayer());
  ohm::Voxel<ohm::CovarianceVoxel> cov_layer_(map_.get(), map_->layout().covarianceLayer());
  ohm::Voxel<ohm::IntensityMeanCov> intensity_layer_(map_.get(), map_->layout().intensityLayer());
  ohm::Voxel<ohm::HitMissCount> hm_layer_(map_.get(), map_->layout().hitMissCountLayer());
  ohm::Voxel<ohm::VoxelSecondarySample> second_return_layer_(map_.get(), map_->layout().secondarySamplesLayer());
  ohm::Voxel<float> trav_layer(map_.get(), map_->layout().traversalLayer());
  ohm::Voxel<ohm::AppearanceVoxel> rgb_layer(map_.get(), map_->layout().appearanceLayer());

  ohm::VoxelMean mean_voxel;
  ohm::HitMissCount hm_voxel;
  ohm::VoxelSecondarySample sr_voxel;
  ohm::AppearanceVoxel rgb_voxel;

  /// Check if layers are valid for our configuration
  for (size_t i = 0; i < msg->width; i++)
  { 
    const Eigen::Vector3d pos_m = T_m_message * Eigen::Vector3d(*pos_iter.x_, *pos_iter.y_, *pos_iter.z_);
    const ohm::Key key = map_->voxelKey(glm::dvec3(pos_m.x(), pos_m.y(), pos_m.z()));

    /// Set all the layers
    ohm::setVoxelKey(key, mean_layer_, occ_layer_, intensity_layer_, hm_layer_, cov_layer_, second_return_layer_,trav_layer,rgb_layer );

    /// Occ voxel. Note that we maintain the log-odds probability here!
    occ_layer_.write(*occ_iter.occ_log_);

    ohm::setPositionSafe(mean_layer_, glm::dvec3(pos_m.x(), pos_m.y(), pos_m.z()), *occ_iter.count_);

    // Cov
    if (params_.ndt_tm_enable)
    {

      // Covar
      ohm::CovarianceVoxel cov_voxel;
      cov_voxel.trianglar_covariance[0] = *ndt_iter_ptr->cov_xx_sqrt;
      cov_voxel.trianglar_covariance[1] = *ndt_iter_ptr->cov_xy_sqrt;
      cov_voxel.trianglar_covariance[2] = *ndt_iter_ptr->cov_xz_sqrt;
      cov_voxel.trianglar_covariance[3] = *ndt_iter_ptr->cov_yy_sqrt;
      cov_voxel.trianglar_covariance[4] = *ndt_iter_ptr->cov_yz_sqrt;
      cov_voxel.trianglar_covariance[5] = *ndt_iter_ptr->cov_zz_sqrt;  // Looks fine
      cov_layer_.write(cov_voxel);

      ohm::IntensityMeanCov int_voxel;
      int_voxel.intensity_mean = *ndt_iter_ptr->intensity_mean;
      int_voxel.intensity_cov = *ndt_iter_ptr->intensity_cov;
      intensity_layer_.write(int_voxel);

      hm_voxel.hit_count = *ndt_iter_ptr->hit_count;
      hm_voxel.miss_count = *ndt_iter_ptr->miss_count;
      hm_layer_.write(hm_voxel);

      ndt_iter_ptr->increment();
    }

    /// Traversable information
    if (params_.traversal_enable)
    {
      trav_layer.write(*trav_iter_ptr->traversal);

      trav_iter_ptr->increment();
    }

    if (params_.second_return_enable)
    {
      // // These are the values of the fields of second returns and not the actual values itself
      sr_voxel.m2 = (float(*sr_iter_ptr->sr_var) * float(*sr_iter_ptr->sr_var));
      sr_voxel.range_mean = uint16_t(*sr_iter_ptr->sr_mean * ohm::secondarySampleQuantisationFactor());
      sr_voxel.count = *sr_iter_ptr->sr_count;
      second_return_layer_.write(sr_voxel);

      sr_iter_ptr->increment();
    }

    if(params_.colour_enable){
      rgb_voxel.red[0] =  float(*rgb_iter_ptr->r)  / 255.0;
      rgb_voxel.green[0] =  float(*rgb_iter_ptr->g)  / 255.0;
      rgb_voxel.blue[0] =  float(*rgb_iter_ptr->b)  / 255.0;

      rgb_layer.write(rgb_voxel);
      rgb_iter_ptr->increment();
    }
    pos_iter.increment();
    occ_iter.increment();
  }
}


auto OhmCloudProcessor::checkFields(const sensor_msgs::PointCloud2::ConstPtr msg) -> void
{
  std::unordered_set<std::string> field_name_set;
  for (auto field : msg->fields)
  { 
    field_name_set.insert(field.name);
  }

  this->params_.ndt_tm_enable = field_name_set.count("covariance_xx_sqrt") > 0;
  this->params_.traversal_enable = field_name_set.count("traversal") > 0;
  this->params_.second_return_enable = field_name_set.count("secondary_sample_count") > 0;
  this->params_.colour_enable = field_name_set.count("red") > 0;
};

}  // namespace ohm_ros