#include "PoseCloudExtractorNode.h"
#include <nve_core/CsvWriter.h>

#include <ohm_ros/OhmCloudProcessor.h>
#include <tf2_eigen/tf2_eigen.h>


#include <Eigen/Core>

#include <filesystem>
#include <sstream>

namespace pdag
{

PoseCloudExtractorNode::PoseCloudExtractorNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private):
nh_(nh),
nh_private_(nh_private)
{ 
  nh_private_.param<std::string>("file_dir", params_.file_dir, std::string{"/data/debug"});
  nh_private_.param<std::string>("target_frame", params_.target_frame, std::string{ "map" });
  nh_private_.param<bool>("colour_enable", params_.colour_enable, false);
  nh_private_.param<bool>("second_return_enable", params_.second_return_enable, true);
  nh_private_.param<bool>("traversal_enable", params_.traversal_enable, true);
  nh_private_.param<bool>("ndt_tm_enable", params_.traversal_enable, true);
  nh_private_.param<bool>("colour_enable", params_.colour_enable, true);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
  tf_list_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  ohm_cloud_sub_ = nh_.subscribe("ohm_cloud", 5, &PoseCloudExtractorNode::ohm_cloud_callback, this);
  ROS_INFO_STREAM(" File dir: " << params_.file_dir);

  setup_dir();
  // main_cb_timer_ = nh_.createTimer(ros::Duration(1 / params_.rate), &FeatureMapRos::main_cb, this, false);
}

void PoseCloudExtractorNode::setup_dir()
{
  auto file_dir = std::filesystem::path(params_.file_dir);
  if (not std::filesystem::exists(file_dir))
  {
    std::filesystem::create_directories(file_dir);
  }

  auto pose_cloud_dir = file_dir / "ohm_scans";
  if (not std::filesystem::exists(pose_cloud_dir))
  {
    std::filesystem::create_directories(pose_cloud_dir);
  }

  auto roi_cloud_dir = file_dir / "roi_clouds";
  if (not std::filesystem::exists(roi_cloud_dir))
  {
    std::filesystem::create_directories(roi_cloud_dir);
  }

  auto resutls_cloud_dir = file_dir / "results";
  if (not std::filesystem::exists(resutls_cloud_dir))
  {
    std::filesystem::create_directories(resutls_cloud_dir);
  }
}


auto PoseCloudExtractorNode::ohm_cloud_callback(const sensor_msgs::PointCloud2::ConstPtr &msg) -> void
{
  /// Check what can be processed
  const auto clock_t0 = std::chrono::high_resolution_clock::now();
  this->checkFields(msg);

  /// Get Transform for msg to world frame:
  Eigen::Affine3d T_m_msg = Eigen::Affine3d::Identity();

  if (msg->header.frame_id != params_.target_frame)
  {
    try
    {
      /// We use the  most last transform if tf_look_up_duration is zero
      auto look_up_time = params_.tf_look_up_duration > 0.0 ? msg->header.stamp : ros::Time(0.0);
      auto transformStamped =
        tf_buffer_->lookupTransform(params_.target_frame, msg->header.frame_id, look_up_time,
                                    ros::Duration(params_.tf_look_up_duration));  // TODO: - parameters
      T_m_msg = tf2::transformToEigen(transformStamped);
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("%s", ex.what());
      return;
    }
  }

  processOhmCloud(msg, T_m_msg);

  /// Check the fields and write them to csv
  const auto clock_t1 = std::chrono::high_resolution_clock::now();
  const auto clock_dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t1 - clock_t0).count();
  ROS_INFO_STREAM_THROTTLE(1, "Time of pcl callback: " << clock_dt);
}

void PoseCloudExtractorNode::checkFields(const sensor_msgs::PointCloud2::ConstPtr msg)
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
}


void PoseCloudExtractorNode::processOhmCloud(const sensor_msgs::PointCloud2::ConstPtr msg,
                                             const Eigen::Affine3d &T_m_message)
{
  /// Mean layer
  std::vector<std::string> header{};
  ohm_ros::BaseCloudIterator pos_iter(msg);
  header.push_back("x");
  header.push_back("y");
  header.push_back("z");

  ohm_ros::OccCloudIterator occ_iter(msg);
  header.push_back("mean_count");
  header.push_back("occupancy_log_probability");

  /// Check if unique pointers are slow!
  std::unique_ptr<ohm_ros::NdtCloudIterator> ndt_iter_ptr{};
  if (params_.ndt_tm_enable)
  {
    ndt_iter_ptr = std::make_unique<ohm_ros::NdtCloudIterator>(msg);
    header.push_back("intensity_mean");
    header.push_back("intensity_covariance");
    header.push_back("miss_count");
    header.push_back("hit_count");
    header.push_back("permeability");
    header.push_back("covariance_xx_sqrt");
    header.push_back("covariance_xy_sqrt");
    header.push_back("covariance_xz_sqrt");
    header.push_back("covariance_yy_sqrt");
    header.push_back("covariance_yz_sqrt");
    header.push_back("covariance_zz_sqrt");

  }

  std::unique_ptr<ohm_ros::TraversalCloudIterator> trav_iter_ptr{};
  if (params_.traversal_enable)
  {
    header.push_back("traversal");
    trav_iter_ptr = std::make_unique<ohm_ros::TraversalCloudIterator>(msg);
  }

  std::unique_ptr<ohm_ros::SecondarySampleIterator> sr_iter_ptr{};
  if (params_.second_return_enable)
  {
    header.push_back("secondary_sample_count");
    header.push_back("secondary_sample_range_mean");
    header.push_back("secondary_sample_range_std_dev");
    sr_iter_ptr = std::make_unique<ohm_ros::SecondarySampleIterator>(msg);
  }

  std::unique_ptr<ohm_ros::RGBSampleIterator> rgb_iter_ptr{};
  if (params_.colour_enable)
  {
    header.push_back("red");
    header.push_back("green");
    header.push_back("blue");
    rgb_iter_ptr = std::make_unique<ohm_ros::RGBSampleIterator>(msg);
  }

  /// Check if layers are valid for our configuration
  auto time_stamp = std::to_string(msg->header.stamp.sec) + "_" + std::to_string(msg->header.stamp.nsec);
  auto file_name = std::filesystem::path(params_.file_dir) / "ohm_scans" /
                   ("ohmscan" + time_stamp +".csv");
  nve_core::io::CsvWriter file_streamer(file_name);

  file_streamer.writeHeader(header);
  for (size_t i = 0; i < msg->width; i++)
  {
    const Eigen::Vector3d pos_m = T_m_message * Eigen::Vector3d(*pos_iter.x_, *pos_iter.y_, *pos_iter.z_);
    std::vector<double> data;
    data.reserve(header.size());

    /// XYZ [0,1,2]
    data.emplace_back(pos_m[0]);
    data.emplace_back(pos_m[1]);
    data.emplace_back(pos_m[2]);

    pos_iter.increment();
    /// mean_count, occ_log_probability [3,4]
    data.emplace_back(*occ_iter.count_);
    data.emplace_back(*occ_iter.occ_log_);
    occ_iter.increment();
    // Cov [5, 16]
    if (params_.ndt_tm_enable)
    {
      // Covar
      data.emplace_back(*ndt_iter_ptr->intensity_mean);
      data.emplace_back(*ndt_iter_ptr->intensity_cov);
      data.emplace_back(*ndt_iter_ptr->miss_count);
      data.emplace_back(*ndt_iter_ptr->hit_count);
      data.emplace_back( double(*ndt_iter_ptr->miss_count) / double((*ndt_iter_ptr->hit_count) + ( *ndt_iter_ptr->miss_count)));
      data.emplace_back(*ndt_iter_ptr->cov_xx_sqrt);
      data.emplace_back(*ndt_iter_ptr->cov_xy_sqrt);
      data.emplace_back(*ndt_iter_ptr->cov_xz_sqrt);
      data.emplace_back(*ndt_iter_ptr->cov_yy_sqrt);
      data.emplace_back(*ndt_iter_ptr->cov_yz_sqrt);
      data.emplace_back(*ndt_iter_ptr->cov_zz_sqrt);  // Looks fine

      ndt_iter_ptr->increment();
    }

    /// Traversable information
    if (params_.traversal_enable)
    {
      data.emplace_back(*trav_iter_ptr->traversal);
      trav_iter_ptr->increment();
    }

    if (params_.second_return_enable)
    {
      data.emplace_back(*sr_iter_ptr->sr_count);
      data.emplace_back(*sr_iter_ptr->sr_mean);
      data.emplace_back(*sr_iter_ptr->sr_var);

      sr_iter_ptr->increment();
      // Write to csv
    }

    if(params_.colour_enable)
    {
      data.emplace_back( float(*rgb_iter_ptr->r)  / 255.0);
      data.emplace_back( float(*rgb_iter_ptr->g)  / 255.0);
      data.emplace_back( float(*rgb_iter_ptr->b)  / 255.0);

      rgb_iter_ptr->increment();
    }

    file_streamer.streamFeature(data);
  }
  file_streamer.close();
}
}  // namespace pdag


int main(int argc, char **argv)
{
  ros::init(argc, argv, "PoseCloudExtractorROS");
  ros::NodeHandle nh{};
  ros::NodeHandle private_nh{ "~" };

  pdag::PoseCloudExtractorNode node(nh, private_nh);

  ros::spin();

  return 0;
}
