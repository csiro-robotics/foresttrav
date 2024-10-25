#ifndef FEATURE_MAP_ROS_HPP
#define FEATURE_MAP_ROS_HPP
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tf2_ros/transform_listener.h>


#include <ros/ros.h>
namespace pdag{

class PoseCloudExtractorNode
{ 
public:
  PoseCloudExtractorNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private);

  struct ParamsPCEN
  { std::string file_dir{""};
    std::string target_frame{};
    double tf_look_up_duration{};

    bool second_return_enable{false};
    bool traversal_enable {false};
    bool ndt_tm_enable{false};
    bool colour_enable{false};

    

  } params_;

private:
  /// @brief Callback for the lidar_ohm_cloud
  void ohm_cloud_callback(const sensor_msgs::PointCloud2::ConstPtr &msg);

  /// @brief  Checks the msg header and toggles the point fields
  void checkFields(const sensor_msgs::PointCloud2::ConstPtr msg);

  /// @brief Generates the required file structures at data_dir
  void setup_dir();

  void processOhmCloud(const sensor_msgs::PointCloud2::ConstPtr msg, const Eigen::Affine3d &T_m_message);

  /// ROS parameters
  ros::NodeHandle nh_, nh_private_;  
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_list_;
  ros::Subscriber ohm_cloud_sub_;

  std::vector<std::string> header_{};

};

}  // namespace pdag
#endif