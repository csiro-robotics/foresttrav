#ifndef VOXEL_LABLER_NODE_H
#define VOXEL_LABLER_NODE_H

#include "BinaryTeLablerROS.h"
#include "NveLabler.h"
#include "ProbTeLablerROS.h"
#include "GroundLabeler.h"
#include "nve_core/FeatureExtractor.h"
#include "nve_core/OhmVoxelStatistic.h"

#include <geometry_msgs/PoseArray.h>

#include <pcl_conversions/pcl_conversions.h>


namespace nve_lfe
{

enum LabelStrategy
{
  BinaryTe = 0,
  ProbTe = 1,
  Ground = 2,
};
enum FeatureExtractorStrategy
{
  VoxelStatistics = 0,
  FtmFeatures     = 1,
  FtmAdjFeatures  = 2
};

struct OffVoxelLablerParams
{
  bool is_pipeline{false};
  bool use_hl{ false }; // Use  hand-labeled data
  std::vector<std::string> hl_files{}; // list of absolut paths of files for the hand-labelled data

  bool use_lfe{ false }; // Enable learning from experience
  
  bool save_lfe_to_file;  // Save the lfe data to file
  std::string out_lfe_cloud_file{}; // Path to the output lfe cloud file

  /// Flags for Voxelfeature extraction
  bool extract_features{ false };
  std::string out_dir{};
  bool extract_te_features{ false };
  bool extract_nte_features{ false };

  LabelStrategy label_strategy{};  
  nve_core::AdjMode_t adj_mode{};

  bool requires_lfe_data{ false }; // Require lfe to label a voxel
  int min_voxel_count{ 4 }; // Require a minimum number of voxels to label a voxel

  // Debug params
  bool visualize{false};
};


class OfflineVoxelLablerNode
{
public:
  OfflineVoxelLablerNode(ros::NodeHandle &nh, ros::NodeHandle &private_nh);
  ~OfflineVoxelLablerNode() = default;

  auto process() -> void;

  bool is_pipeline(){return params_.is_pipeline;};

private:
  ros::NodeHandle nh_, private_nh_;
  std::shared_ptr<ohm::OccupancyMap> map_;
  std::shared_ptr<nve_lfe::Labler> labler_;
  std::shared_ptr<nve_core::FeatureExtractorBase> extractor_;

  /// @brief Loads the ros params
  void loadRosParams();

  /// @brief Sets the labeling strategt
  auto setLabelStrategy(const LabelStrategy label_strategy) -> void;

  /// @brief
  auto setExtractorStrategy(const FeatureExtractorStrategy feature_extractor_strategy)->void;

  /// @brief Integrates the hand labelled data into the map prior to the trajectory
  /// Note: We view the hand labeled data as a map-prior
  auto processHandLabelledData() -> void;

  /// @brief Fuses the semantic trajectory into the map reprsentation, based on the strategt
  auto processSemanticTrajectory() -> void;

  /// @brief 
  /// @return 
  auto extractPerVoxelFeatures() -> void;

  /// @brief 
  /// @param key_list 
  /// @param label 
  /// @return 
  auto extractAndWriteFeatureClass(const std::vector<ohm::Key> &key_list, const TeLabel label)
    -> void;

  /// @brief 
  /// @return 
  auto generateDebugCloud() -> void;

  auto debug_cb(const ros::TimerEvent &) -> void;
  
  auto setup_dir() -> void;

  OffVoxelLablerParams params_;
  nve_tools::TrajectoryIO semantic_traj_loader_{};

  geometry_msgs::PoseArray poses_{};
  std::vector<double> map_roi_{};
  pcl::PointCloud<pcl::PointXYZRGBL> cloud_{};

  ros::Publisher debug_pub_lfe_cloud_;
  ros::Publisher debug_pub_lfe_poses_;
  ros::Timer debug_timer_;
};

}  // namespace nve_lfe
#endif