#ifndef OHM_COLOUR_FUSER_ROS_H
#define OHM_COLOUR_FUSER_ROS_H

#include <nve_core/ColourFuser.h>

#include <ohm/ohm/OccupancyMap.h>
#include <ohm/ohm/RayMapperSecondarySample.h>
#include <ohm/KeyRange.h>

#include <sensor_msgs/PointCloud2.h>
// #include <std_srvs/Empty.h>
#include <tf2_ros/transform_listener.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>

namespace nve_ros
{
class OhmColourFuserRos
{
public:
  explicit OhmColourFuserRos(ros::NodeHandle &nh, ros::NodeHandle &nh_private);


private:
  /// @brief Main callback loop 
  /// @param[in] void Timer for main callback loop 
  auto main_cb(const ros::TimerEvent &) -> void;

  /// @brief Callback for incoming colour cloud
  auto colour_cloud_cb(const sensor_msgs::PointCloud2::ConstPtr &msg)->void;

  /// @brief Callback for incoming colourised point cloud which encode rgba as single float
  auto colour_cloud_compressed_cb(const sensor_msgs::PointCloud2::ConstPtr &msg) -> void;

  /// Processes the 
  auto processed_compressed_cloud(const sensor_msgs::PointCloud2::ConstPtr &msg) -> void;

  /// @brief callback timer for processing new clearing keys/points
  /// TODO: Implement
  auto clear_cb(const ros::TimerEvent &) -> void;

  /// @brief Callback for incoming clearing points
  /// TODO: Implement
  auto clear_cloud_cb(const sensor_msgs::PointCloud2::ConstPtr &msg)->void;
  
  /// @brief updates local posisiont
  auto get_latest_position(glm::dvec3 &latest_pos) -> bool;

  auto rgbcloud_from_aabb(pcl::PointCloud<pcl::PointXYZRGB> &cloud) -> bool;

  ros::NodeHandle nh_, nh_private_;
  ros::Timer main_cb_timer_;
  ros::Publisher rgb_cloud_pub_;
  ros::Subscriber sub_rgb_cloud_;
  ros::Subscriber sub_rgb_cloud_compressed_;

  std::vector<double> map_bounds_{};

  std::shared_ptr<ohm::OccupancyMap> map_;  /// Map instance
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_list_;

  std::shared_ptr<nve_core::ColourFuserEP> fuser_;

  bool map_updated_{false};
  bool use_multi_thread_{false};

  std::string world_frame_{};
  std::string robot_frame_{};
  std::string camera_frame_{};

};



}
#endif  //OHM_COLOUR_FUSER_ROS_H