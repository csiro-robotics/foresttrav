// Copyright (c) 2024
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz
#ifndef OHM_COSTMAP_GENERATOR_H
#define OHM_COSTMAP_GENERATOR_H

#include "GroundMapInterface.h"

#include <geometry_msgs/Pose.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <tf2_ros/transform_listener.h>

#include <ros/ros.h>

#include <mutex>


/// Notes: The incoming pcl is filtered on arrival (not transformed). The robot bound (particularly in z directin) can
/// alter results if the ground points and robot bounds are to tight.
/// TODO: Should the incoming pcl be filtered?

namespace nve_ros
{

/// @brief Finds the lowest occupied voxel for a region of interrest
/// @param map List of keys wich make up the ground
/// @param ground_map The associated ground cost
/// @brief Generates the debug ground cloud at -10 meters under the surface
auto generateDebugGroundCloud(ohm::OccupancyMap &map,  std::unordered_map<ohm::Key, Cell> const &ground_map,
                              pcl::PointCloud<pcl::PointXYZI> &cloud) -> void;

/// @brief Checks if a point is within bounds
/// @param point Point to check, expected a double
/// @param bounds Bounds for the point to lie in
/// @return True if the point lies within, false otherwise
template <typename T>
auto isWithinBounds( T const &point,  std::vector<double> const &bounds) -> bool;

class Ohm2dCostmapGenerator
{
public:
  Ohm2dCostmapGenerator(ros::NodeHandle &nh, ros::NodeHandle &nh_private);

  auto loadParams()-> void;

  /// @brief Main callback loop
  auto mainCb(ros::TimerEvent const &)-> void;

private:
  struct Params
  {
    double rate{ 5.0 };                        /// Rate at which the loop runs at [hz]
    std::string world_frame{ "odom" };         /// Static world frame
    std::string robot_frame{ "base_link" };    /// Robot frame

    double map_res{ 0.0 };  /// map resolution
    std::vector<double> local_map_bounds{ -1.0, -1.0, -3.0, 1.0, 1.0, 3 };

    bool has_new_te_map{ false };  /// Flag to determin the arrival of new data

    double tf_look_up_duration{ 0.0 };  /// Time to lookup
    int num_vertical_voxels{ 10 };      /// Number of voxel above ground voxel

    bool use_virtual{ false };  // If virtual surfaces should be used
    bool use_patch_fill{
      false
    };  // Adds virtual ground around a ground cell. The height and cost are the mean average of
    int patch_leaf_size{ 3 };  // Defines the patch size. Total size is 2n+2. E.x 3x3 is patch size 1, 5x5 patch size 2
    int min_adj_voxel{ 3 };    // The minimum amount of adjacent voxel that need to be considered for a patch operation
    bool use_patch_flood_fill{ false };  // Flag to enable a flodd fill algorithem.
  };

  /// @brief Incoming cloud from the traversability estimation (does not change frame of cloud)
  /// @param cloud_msg Incoming te estimation [x,y,z, prob_te, label_te] currently
  auto teCloudCb(sensor_msgs::PointCloud2::ConstPtr const &msg) -> void;

  /// @brief Incoming cloud from the traverability estimation node
  /// @param cloud_msg Incoming te estimation [x,y,z, prob_te, label_te] currently
  auto publishCostmap(const sensor_msgs::PointCloud2::ConstPtr &msg) -> void;

  /// @brief If the current 2dcostmap is still valid or needs to be recomputedl
  /// - Either it is not computed already
  /// - Update of the map or costmap i
  auto generateCostmap(const std::vector<ohm::Key> &ground_keys, std::vector<double> &ground_cost) -> void;

  /// @brief Mean collum costing, where the cost is the mean value of the tp of the voxel_num above the ground voxe, if
  /// they contain data
  auto MeanColumnCosting(int voxel_num) -> void;

  /// @brief Generates the missing ground keys for the ROI
  /// @param roi   Region of interest in the global frame
  auto generateVirtualCost(std::vector<double> const &roi, size_t patch_leaf_size) -> void;

  /// @brief Generated the bounds considering the pose of the robot and the bounding box dimensions
  /// @return Bounds x_min,y_min,.. ,z_max
  auto getCurrentBounds() -> std::vector<double>;

  ros::NodeHandle nh_, nh_private_;
  ros::Timer main_timer_;
  ros::Subscriber sub_ohm_mp_cb_, te_pcl_cb_;
  ros::Publisher costmap_pub_, dense_costmap_pub_;

  Params params_;

  std::shared_ptr<ohm::OccupancyMap> map_;  /// Map instance
  std::shared_ptr<nve_ros::GroundMapInterface> ground_mapper_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_list_;
  std::string msg_frame_id_{};

  std::mutex mutex_;
};


}  // namespace nve_ros
#endif  // OHM_COLOUR_FUSER_ROS_H
