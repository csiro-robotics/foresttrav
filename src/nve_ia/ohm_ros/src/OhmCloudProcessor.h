#ifndef OHM_CLOUD_HELPER_H
#define OHM_CLOUD_HELPER_H

#include <ohm/OccupancyMap.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <unordered_set>

#include <Eigen/Geometry>

namespace ohm_ros
{

/// @brief: The ohmCloudisPacked as dense cloud.
class OhmCloudProcessor
{
public:
  struct OhmCloudOptions
  {
    bool ndt_tm_enable{ false };
    bool traversal_enable{ false };
    bool colour_enable{ false };
    bool second_return_enable{ false };
  };

  explicit OhmCloudProcessor(std::shared_ptr<ohm::OccupancyMap> map)
    : map_(map){};

  OhmCloudProcessor(std::shared_ptr<ohm::OccupancyMap> map, OhmCloudOptions params)
    : map_(map), params_(params){};

  void processOhmCloud(const sensor_msgs::PointCloud2::ConstPtr msg,  Eigen::Affine3d &T_m_message);

  void updateParams(const OhmCloudOptions params){params_ = params;};

  void checkFields(const sensor_msgs::PointCloud2::ConstPtr msg);
  
  void enableNDT(){params_.ndt_tm_enable = true;};
  void enableColour(){params_.colour_enable = true;};
  void enableSecondReturns(){params_.second_return_enable = true;};
  void enableTaversal(){params_.traversal_enable = true;};

private:
  
  OhmCloudOptions params_;
  std::shared_ptr<ohm::OccupancyMap> map_{};

  /// @brief checks the fields of a msg and
};


//// Iterators to simplify the interface for processing an incoming point cloud with the ohm features
struct BaseCloudIterator
{
  explicit BaseCloudIterator(const sensor_msgs::PointCloud2::ConstPtr msg)
    : x_(*msg, "x")
    , y_(*msg, "y")
    , z_(*msg, "z"){};
  sensor_msgs::PointCloud2ConstIterator<float> x_;
  sensor_msgs::PointCloud2ConstIterator<float> y_;
  sensor_msgs::PointCloud2ConstIterator<float> z_;

  void increment()
  {
    ++x_;
    ++y_;
    ++z_;
  };
};

struct OccCloudIterator
{
  explicit OccCloudIterator(const sensor_msgs::PointCloud2::ConstPtr msg)
    : count_(*msg, "mean_count")
    , occ_log_(*msg, "occupancy_log_probability"){};
  sensor_msgs::PointCloud2ConstIterator<uint32_t> count_;
  sensor_msgs::PointCloud2ConstIterator<float> occ_log_;

  void increment()
  {
    ++count_;
    ++occ_log_;
  };
};

struct TraversalCloudIterator
{
  explicit TraversalCloudIterator(const sensor_msgs::PointCloud2::ConstPtr msg)
    : traversal(*msg, "traversal"){};
  sensor_msgs::PointCloud2ConstIterator<float> traversal;

  void increment() { ++traversal; };
};

struct NdtCloudIterator
{
  explicit NdtCloudIterator(const sensor_msgs::PointCloud2::ConstPtr msg):
      intensity_mean(*msg, "intensity_mean")
    , intensity_cov(*msg, "intensity_covariance")
    , cov_xx_sqrt(*msg, "covariance_xx_sqrt")
    , cov_xy_sqrt(*msg, "covariance_xy_sqrt")
    , cov_xz_sqrt(*msg, "covariance_xz_sqrt")
    , cov_yy_sqrt(*msg, "covariance_yy_sqrt")
    , cov_yz_sqrt(*msg, "covariance_yz_sqrt")
    , cov_zz_sqrt(*msg, "covariance_zz_sqrt")
    , hit_count(*msg, "hit_count")
    , miss_count(*msg, "miss_count"){};
  sensor_msgs::PointCloud2ConstIterator<float> cov_xx_sqrt;
  sensor_msgs::PointCloud2ConstIterator<float> cov_xy_sqrt;
  sensor_msgs::PointCloud2ConstIterator<float> cov_xz_sqrt;
  sensor_msgs::PointCloud2ConstIterator<float> cov_yy_sqrt;
  sensor_msgs::PointCloud2ConstIterator<float> cov_yz_sqrt;
  sensor_msgs::PointCloud2ConstIterator<float> cov_zz_sqrt;

  sensor_msgs::PointCloud2ConstIterator<float> intensity_mean;
  sensor_msgs::PointCloud2ConstIterator<float> intensity_cov;
  sensor_msgs::PointCloud2ConstIterator<uint32_t> hit_count;
  sensor_msgs::PointCloud2ConstIterator<uint32_t> miss_count;

  void increment()
  {
    ++cov_xx_sqrt;
    ++cov_xy_sqrt;
    ++cov_xz_sqrt;
    ++cov_yy_sqrt;
    ++cov_yz_sqrt;
    ++cov_zz_sqrt;

    ++intensity_mean;
    ++intensity_cov;
    ++hit_count;
    ++miss_count;
  };
};

struct SecondarySampleIterator
{
  explicit SecondarySampleIterator(const sensor_msgs::PointCloud2::ConstPtr msg)
    : sr_mean(*msg, "secondary_sample_range_mean")
    , sr_var(*msg, "secondary_sample_range_std_dev")
    , sr_count(*msg, "secondary_sample_count"){};
  
  /// Internal iterators
  sensor_msgs::PointCloud2ConstIterator<double> sr_mean;
  sensor_msgs::PointCloud2ConstIterator<double> sr_var;
  sensor_msgs::PointCloud2ConstIterator<uint16_t> sr_count;

  void increment()
  {
    ++sr_mean;
    ++sr_var;
    ++sr_count;
  };
};

struct RGBSampleIterator
{
  explicit RGBSampleIterator(const sensor_msgs::PointCloud2::ConstPtr msg)
    : r(*msg, "r")
    , g(*msg, "g")
    , b(*msg, "b"){};
  
  /// Internal iterators
  sensor_msgs::PointCloud2ConstIterator<uint8_t> r;
  sensor_msgs::PointCloud2ConstIterator<uint8_t> g;
  sensor_msgs::PointCloud2ConstIterator<uint8_t> b;

  void increment()
  {
    ++r;
    ++g;
    ++b;
  };
};


}  // namespace ftm_ros

#endif