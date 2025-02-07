#include "OfflineVoxelLablerNode.h"
#include "lfe_utils.h"
#include "nve_core/CsvWriter.h"
#include "nve_core/FeatureExtractor.h"
#include "tinycolourmap.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <slamio/PointCloudReaderPly.h>

#include <filesystem>

namespace nve_lfe
{

/// TODO:
/// Use either FeatureCloudGenerator or OhmStatisticsExtractor

OfflineVoxelLablerNode::OfflineVoxelLablerNode(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
  : nh_(nh)
  , private_nh_(private_nh)
{
  loadRosParams();

  if (true)
  {
    debug_pub_lfe_cloud_ = private_nh_.advertise<pcl::PointCloud<pcl::PointXYZRGBL>>("lfe_cloud", 1);
    debug_pub_lfe_poses_ = private_nh_.advertise<geometry_msgs::PoseArray>("lfe_poses", 1);
    debug_timer_ = private_nh_.createTimer(ros::Duration(1), &OfflineVoxelLablerNode::debug_cb, this, false);
  }
}

auto OfflineVoxelLablerNode::process() -> void
{
  // Extract and add the hand-labelled data
  // Note that the fusion strategy is set by the labler strategy
  processHandLabelledData();

  processSemanticTrajectory();

  extractPerVoxelFeatures();

  generateDebugCloud();
}


void OfflineVoxelLablerNode::loadRosParams()
{
  // Load semantic trajectory
  std::string semantic_traj_file{};
  private_nh_.param<std::string>("semantic_traj_file", semantic_traj_file, "");
  if (not semantic_traj_loader_.load(semantic_traj_file))
  {
    ROS_FATAL_STREAM("Failed to load Semantic Trajectory file: " << semantic_traj_file);
    return;
  }

  /// Load OHM map
  std::string map_file{};
  private_nh_.param<std::string>("map_file", map_file, "");
  map_ = std::make_shared<ohm::OccupancyMap>(0.1);
  if (ohm::load(map_file.c_str(), *map_, nullptr, nullptr))
  {
    ROS_FATAL_STREAM("FAILED TO LOAD OHM FILE " << map_file);
    return;
  }

  /// Setup semantic layer
  auto semantic_layer_id = map_->layout().semanticLayer();
  if (0 > semantic_layer_id)
  {
    ohm::MapLayout new_layout = map_->layout();
    ohm::addSemanticLayer(new_layout);
    map_->updateLayout(new_layout);
  }


  /// Setup labler
  int label_strategy;
  private_nh_.param("use_lfe", params_.use_lfe, true);
  private_nh_.param("use_hl", params_.use_hl, true);
  private_nh_.param("hl_files", params_.hl_files, std::vector<std::string>{ "" });
  ROS_INFO("Number of hl files %i", params_.hl_files.size());
  private_nh_.param("label_strategy", label_strategy, int{ 0 });
  private_nh_.param("extract_te_features", params_.extract_te_features, true);
  private_nh_.param("extract_nte_features", params_.extract_nte_features, true);

  setLabelStrategy(static_cast<LabelStrategy>(label_strategy));

  int feature_strategy{ 0 };
  private_nh_.param("feature_strategy", feature_strategy, int{ 0 });

  setExtractorStrategy(static_cast<FeatureExtractorStrategy>(feature_strategy));


  /// IO configurations
  private_nh_.param("out_dir", params_.out_dir, std::string{ "" });
  if (params_.out_dir.empty())
  {
    ROS_FATAL_STREAM("out_dir is empty");
    return;
  }
  
  // Setup the data_root directories
  if (!std::filesystem::create_directories(params_.out_dir))
  {
    if (!std::filesystem::exists(params_.out_dir))
    {
      ROS_FATAL_STREAM("Failed to create out_dir: " << params_.out_dir);
      return;
    }
  }

  /// MAP ROI
  private_nh_.param("map_roi", map_roi_, std::vector<double>{ -10.0, -10.0, -0.1, 10.0, 10.0, 0.1 });
  ROS_INFO("Map_roi: %f, %f, %f, %f, %f, %f", map_roi_[0], map_roi_[1], map_roi_[2], map_roi_[3], map_roi_[4],
           map_roi_[5]);
  private_nh_.param("is_pipeline", params_.is_pipeline, false);
  ROS_INFO("Finish loading parameters");
}

auto OfflineVoxelLablerNode::processSemanticTrajectory() -> void
{
  if (not params_.use_lfe)
  {
    ROS_INFO("Skipping Lfe labeling");
    return;
  }
  auto traj_points = semantic_traj_loader_.points();
  auto traj_quats = semantic_traj_loader_.quats();
  auto traj_labels = semantic_traj_loader_.labels();
  auto traj_times = semantic_traj_loader_.times();

  if (!(traj_points.size() == traj_quats.size() && traj_points.size() == traj_labels.size() &&
        traj_points.size() == traj_times.size()) or
      ((not traj_points.size()) > 0))
  {
    ROS_FATAL("Semantic Trajectory has not loaded correctly");
    return;
  }


  double last_update_time = 0.0;
  double min_update_interval{};  /// Require half a second for an update
  private_nh_.param<double>("min_update_interval", min_update_interval, 0.5);

  // TODO: CHeck poses for non traverable cases
  for (size_t i = 0; i < traj_labels.size(); i++)
  {
    /// Minimum delta t between two updates
    auto delta_update = traj_times[i] - last_update_time;
    if (delta_update < min_update_interval)
    {
      continue;
    }
    last_update_time = traj_times[i];

    /// Get pose from intance i
    Eigen::Translation3d trans_k = Eigen::Translation3d(traj_points[i][0], traj_points[i][1], traj_points[i][2]);
    Eigen::Quaterniond quat_k =
    Eigen::Quaterniond(traj_quats[i].w, traj_quats[i].x, traj_quats[i].y, traj_quats[i].z);
    quat_k.normalize();
    Eigen::Affine3d T_mr = Eigen::Isometry3d(trans_k * quat_k);

    /// TODO: @p non_valid Non-valid labels skip
    if (traj_labels[i] == -1)
    {
      continue;
    }

    /// Mapping from file labels to TeLabels
    /// TODO: Mapping function and not just adding one
    labler_->processEvent(T_mr, static_cast<nve_lfe::TeLabel>(traj_labels[i] + 1), nullptr, nullptr);

    /// DEBUG: This is to visualize all the collision poses
    if (traj_labels[i] + 1 == nve_lfe::TeLabel::Collision or traj_labels[i] + 1 == nve_lfe::TeLabel::Free)
    {
      geometry_msgs::Pose pose;
      pose.position.x = T_mr.translation().x();
      pose.position.y = T_mr.translation().y();
      pose.position.z = T_mr.translation().z();
      Eigen::Quaterniond q(T_mr.rotation());
      pose.orientation.x = q.x();
      pose.orientation.y = q.y();
      pose.orientation.z = q.z();
      pose.orientation.w = q.w();
      poses_.poses.push_back(pose);
    }
  }
}


void OfflineVoxelLablerNode::generateDebugCloud()
{
  auto min_key = map_->voxelKey(glm::dvec3(map_roi_[0], map_roi_[1], map_roi_[2]));
  auto max_key = map_->voxelKey(glm::dvec3(map_roi_[3], map_roi_[4], map_roi_[5]));
  auto key_range = ohm::KeyRange(min_key, max_key, map_->regionVoxelDimensions());

  /// Colourisation for RGBA
  ohm::SemanticLabel label{};

  ohm::Voxel<ohm::VoxelMean> mean_layer(map_.get(), map_->layout().meanLayer());
  for (auto &key : key_range)
  {
    if (not labler_->getSemanticData(key, label))
    {
      continue;
    }
    /// If occupied or has n measurements
    ohm::setVoxelKey(key, mean_layer);
    ohm::VoxelMean voxel_data;
    mean_layer.read(&voxel_data);

    if (voxel_data.count < 3)
    {
      continue;
    }

    pcl::PointXYZRGBL pcl_point;
    auto pos = map_->voxelCentreGlobal(key);

    pcl_point.x = pos.x;
    pcl_point.y = pos.y;
    pcl_point.z = pos.z;

    /// Colour transform from [0-1] to [0 - 255] in RGB
    const tinycolormap::Color colour =
      tinycolormap::GetColor(nve_lfe::probFromLogsOdds(label.prob_label), tinycolormap::ColormapType::Viridis);

    pcl_point.r = colour.r() * 255;
    pcl_point.g = colour.g() * 255;
    pcl_point.b = colour.b() * 255;
    pcl_point.label = label.label;
    cloud_.emplace_back(pcl_point);
  }

  /// Setup publisher
  ROS_INFO("Finished genearting pcl");
}


auto OfflineVoxelLablerNode::processHandLabelledData() -> void
{
  std::string tag{ "processHandLabelledData" };
  if (not params_.use_hl)
  {
    ROS_WARN_NAMED(tag, "No hand labelled data processed");
    return;
  }

  if ( params_.hl_files.empty())
  {
    ROS_WARN_NAMED(tag, "No hand labelled data processed since no files are found");
    return;
  }

  ROS_WARN_NAMED(tag, "Start hand-label processing");
  for (const auto &file : params_.hl_files)
  {
    slamio::PointCloudReaderPly cloud_loader;
    cloud_loader.setDesiredChannels(slamio::DataChannel::Position | slamio::DataChannel::Label);
    if (!cloud_loader.open(file.c_str()))
    {
      ROS_ERROR_STREAM("Could not load files: " << file);
    }

    slamio::CloudPoint sample_point;
    while (cloud_loader.readNext(sample_point))
    {
      const auto key = map_->voxelKey(sample_point.position);
      labler_->addMapPrior(key, static_cast<TeLabel>(sample_point.label + 1));
    }
    cloud_loader.close();
  }
}

auto OfflineVoxelLablerNode::extractPerVoxelFeatures() -> void
{
  /// TODO: OHM_UTILS: keyRangeFromAABBAroundPose(map, pos, roi/aabb)
  auto min_key = map_->voxelKey(glm::dvec3(map_roi_[0], map_roi_[1], map_roi_[2]));
  auto max_key = map_->voxelKey(glm::dvec3(map_roi_[3], map_roi_[4], map_roi_[5]));
  auto key_range = ohm::KeyRange(min_key, max_key, map_->regionVoxelDimensions());

  auto te_key_list = std::vector<ohm::Key>{};
  auto nte_key_list = std::vector<ohm::Key>{};

  ohm::SemanticLabel label{};
  ohm::Voxel<ohm::VoxelMean> mean_layer(map_.get(), map_->layout().meanLayer());
  /// This is an expensive loop that checks a) That the voxel has lfe data AND some measurements, where
  /// the number 4 is arbitrary
  for (auto &key : key_range)
  {
    /// TODO: OHM_UTILS: getSemanticData(map, key, label)
    if ((not labler_->getSemanticData(key, label)))
    {
      continue;
    }
    /// If occupied or has n measurements and
    ohm::setVoxelKey(key, mean_layer);
    if ((mean_layer.data().count < params_.min_voxel_count) and (params_.min_voxel_count > 0))
    {
      continue;
    }

    // Note: We are seperating the labels based on probability!
    if (nve_lfe::probFromLogsOdds(label.prob_label) >= 0.5)
    {
      te_key_list.emplace_back(key);
    }
    else if (nve_lfe::probFromLogsOdds(label.prob_label) < 0.5)
    {
      nte_key_list.emplace_back(key);
    }
  }

  if (params_.extract_nte_features)
  {
    ROS_INFO_STREAM("Starting class Non-traversable");
    extractAndWriteFeatureClass(nte_key_list, TeLabel::Collision);
  }

  if (params_.extract_te_features)
  {
    ROS_INFO_STREAM("Starting class traversable");
    extractAndWriteFeatureClass(te_key_list, TeLabel::Free);
  }
}

auto OfflineVoxelLablerNode::extractAndWriteFeatureClass(const std::vector<ohm::Key> &key_list, const TeLabel label)
  -> void
{
  std::string file_name = "semantic_cloud_class_" + std::to_string(label) + ".csv";
  std::string out_file = std::filesystem::path(params_.out_dir) / file_name;

  ROS_INFO_STREAM("Writting examples of label " << label << " to :" << out_file);

  auto header_neg = extractor_->getFeatureHeader();

  nve_core::io::CsvWriter csv_writer(out_file);
  csv_writer.writeHeader(header_neg);

  std::vector<double> feature_vector{};


  auto label_id = extractor_->getFeatureId("label");
  if (label_id < 0 || label_id > 50)
  {
    ROS_FATAL_STREAM("Label does not exist in the ohm map or <feature_id> is wrong");
    return;
  }

  for (auto &key : key_list)
  {
    if (extractor_->extractFeatureByKey(key, feature_vector))
    {
      csv_writer.streamFeature(feature_vector);
    }
  }
  csv_writer.close();
  ROS_INFO("Finish processing the feature map");
}


auto OfflineVoxelLablerNode::setLabelStrategy(const LabelStrategy label_strategy) -> void
{
  switch (label_strategy)
  {
  case LabelStrategy::BinaryTe:
    labler_ = std::make_shared<BinaryTeLablerROS>(nh_, private_nh_, map_);
    break;

  case LabelStrategy::ProbTe:
    labler_ = std::make_shared<ProbTeLablerROS>(nh_, private_nh_, map_);
    break;

  case LabelStrategy::Ground:
    labler_ = std::make_shared<GroundLablerROS>(nh_, private_nh_, map_);
    break;

  default:
    labler_ = std::make_shared<ProbTeLablerROS>(nh_, private_nh_, map_);
    break;
  }
}

auto OfflineVoxelLablerNode::setExtractorStrategy(const FeatureExtractorStrategy feature_extractor_strategy) -> void
{
  // switch (feature_extractor_strategy)
  // {
  // case FeatureExtractorStrategy::VoxelStatistics: {
  //   extractor_ = std::make_shared<nve_core::OhmVoxelStatistic>(map_);
  //   break;
  // }
  // case FeatureExtractorStrategy::FtmFeatures: {
  //   extractor_ = std::make_shared<nve_core::FeatureExtractor>(map_);
  //   break;
  // }

  // default: {
  //   extractor_ = std::make_shared<nve_core::OhmVoxelStatistic>(map_);
  //   break;
  // }
  // }
  extractor_ = std::make_shared<nve_core::FeatureExtractor>(map_);

}


auto OfflineVoxelLablerNode::debug_cb(const ros::TimerEvent &) -> void
{
  if (cloud_.points.size() > 0)
  {
    cloud_.header.frame_id = "map";
    pcl_conversions::toPCL(ros::Time::now(), cloud_.header.stamp);
    debug_pub_lfe_cloud_.publish(cloud_);
  }
  if (not poses_.poses.empty())
  {
    poses_.header.frame_id = "map";
    poses_.header.stamp = ros::Time::now();
    debug_pub_lfe_poses_.publish(poses_);
  }
}

auto OfflineVoxelLablerNode::setup_dir() -> void
{
  std::filesystem::path out_dir(params_.out_dir);

  if (!std::filesystem::create_directories(out_dir))
  {
    ROS_FATAL("COULD NOT CREATE FILE DIR");
  }
}


}  // namespace nve_lfe
