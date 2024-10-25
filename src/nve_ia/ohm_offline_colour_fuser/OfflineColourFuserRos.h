#ifndef OFFLINE_COLOUR_FUSER_ROS_H
#define OFFLINE_COLOUR_FUSER_ROS_H


#include <nve_core/NdtApperanceRayFuser.h>
#include <nve_core/RayBuffer.h>
#include <nve_core/VisabilityAssesser.h>

#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>

#include <nve_tools/ImageTSLoader.h>
#include <nve_tools/SemanticDataLoader.h>
#include <nve_tools/TrajectoryLoader.h>

#include <ohm/KeyRange.h>
#include <ohm/ohm/DefaultLayer.h>
#include <ohm/ohm/Key.h>
#include <ohm/ohm/KeyList.h>
#include <ohm/ohm/MapChunk.h>
#include <ohm/ohm/MapLayer.h>
#include <ohm/ohm/MapLayout.h>
#include <ohm/ohm/MapSerialise.h>
#include <ohm/ohm/NdtMap.h>
#include <ohm/ohm/OccupancyMap.h>
#include <ohm/ohm/Voxel.h>
#include <ohm/ohm/VoxelAppearance.h>
#include <ohm/ohm/VoxelData.h>
#include <ohm/slamio/SlamCloudLoader.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl_conversions/pcl_conversions.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <tf2_eigen/tf2_eigen.h>

#include <geometry_msgs/PoseArray.h>

#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

#include <ros/ros.h>

namespace nve_ros
{
/// Forward declarations

class OhmOfflineColourFuserNode
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OhmOfflineColourFuserNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private);

  void load_ros_params();

  void main_cb(const ros::TimerEvent &);

  bool pipeline(){return pipeline_;};

  int current_frame(){return img_idx_;};

  int get_total_number_frames(){return cap_.get(cv::CAP_PROP_FRAME_COUNT);};

  /// Return the robot pose based on a time stamp

  /// @brief Processes the next frame, returns true if the end-of-video is reached
  bool process_next_frame();

  /// @brief Saves the colourized map to a file defined by out_file
  void save_colourized_map();

private:
  /// @brief Get the current robot pose (T_M_R) from map to base_link
  Eigen::Affine3d robotPose(const double timestamp);

  /// @brief Updates all the internal transforms
  void update_transforms(const double timestamp);

  /// @brief publish the colorized image
  void publish_img();

  /// @brief Publish colorized point cloud
  void publish_colorized_points(const std::vector<Eigen::Vector3d> &points,
                                const std::vector<Eigen ::Vector3d> &colours);

  
  /// @brief Debug point cloud to see what points are considered visible
  void publish_visible_points(const std::vector<Eigen::Vector3d> &points);

  /// Params
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  bool pipeline_{};
  bool debug_{};
  bool end_of_video_ = false;
  double rate_;
  ohm::OccupancyMap map_;

  // General Parameters
  nve_tools::TrajectoryIO traj_;
  nve_core::RayBuffer ray_buffer_;
  nve_core::NdtApperanceRayFuser fuser_;
  nve_core::ColourFusionMode mode_{};
  double beta_{};
  nve_core::VisabilityChecker vis_check_;

  /// Video I/O
  cv::VideoCapture cap_;  /// Video capture stream
  cv::Mat img_;           /// Current image
  int img_idx_;           /// Image index
  int step_size_;         /// Size

  std::vector<double> camera_matrix_{};      // fx, fy, cx, cy
  std::vector<double> camera_dist_coeff_{};  // OpenCv definitions
  std::vector<double> image_ts_{};
  cv_bridge::CvImagePtr cv_ptr_;

  // All points which lie within the camera frustum expressed in the camera image frame
  // Assumptions are made on world_frame: map, robot_frame: base_link, etc..
  Eigen::Affine3d T_r_c_{};  // Camera transformation from camera to robot frame
  Eigen::Affine3d T_m_r_{};
  Eigen::Affine3d T_m_c_{};
  Eigen::Affine3d T_c_m_{};

  /// ROS2 parameters
  ros::Timer main_timer_cb_;
  ros::Timer debug_timer_cb_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  image_transport::ImageTransport image_transport_;
  image_transport::Publisher pub_img_;
  ros::Publisher local_pcl_pub_{};
  ros::Publisher full_pcl_pub_{};
  ros::Publisher debug_visible_points_pub_{};

  sensor_msgs::PointCloud2 cloud_{};
  sensor_msgs::PointCloud2 visible_points_{};
  geometry_msgs::PoseArray pose_array_{};
  std::vector<double> map_bounds_{};
  /// Image mask to mask certain areas of the img. [im_w_low, img_w_g, img_h_low, img_h_high] in percentages
  std::vector<double> img_mask_{};

  /// Color pipeline
  bool save_map_ = false;

  // /// Debug functions
  void publish_tf();

  void published_fused_colour_cloud(const ros::TimerEvent &);


  std::string world_frame_{};   ///< default "map"
  std::string robot_frame_{};   //< default" "base_link"
  std::string camera_frame_{};  //< default "cam0"

};

}  // namespace nve_ros
#endif
