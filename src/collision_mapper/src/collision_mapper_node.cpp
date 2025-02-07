/* MIT License
 * Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO) 
 * Queensland University of Technology (QUT)
 *
 * Author: Fabio Ruetz
 */

#include <ros/ros.h>
#include <chrono>

#include <nve_robot_labelling/ProbTeLablerROS.h>
#include <nve_robot_labelling/lfe_utils.h>
#include <nve_robot_labelling/tinycolourmap.h>

#include <ohm/DefaultLayer.h>
#include <ohm/KeyRange.h>
#include <ohm/OccupancyMap.h>
#include <ohm/VoxelSemanticLabel.h>

#include <Eigen/Geometry>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/UInt8.h>
#include <std_srvs/Empty.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

struct CollisionState
{
  ros::Time stamp;
  Eigen::Affine3d pose;
  nve_lfe::TeLabel label;
};


struct CollisionMapParams
{
  std::string world_frame;
  std::string odom_frame;
  std::string robot_frame;

  double voxel_size{ 1.0 };

  double main_cb_rate{ 0.1 };  // Rate at which the main_cb is checked (not processed!)
  double auto_cm_interval{
    1.0
  };  // Time intervale at which the collision map gets automatically computed [s] , <0 no auto compute
  double cs_rate{ 1.0 };           // Rate at which the collision states are sampled
  double dt_pos_cs{ 0.5 };         // time difference for two possive (TE) meassurements
  double min_pose_dist{ 0.5 };     // Minimal distance between two poses [m]
  double dt_collision_reset{ 0 };  // Minimal time [s] that a collision can be recorded

  std::string trigger_topic = "";  // Topic for state of the trigger button, values can range from {0,2}
  int trigger_pressed = 2;         // State assigned to when trigger pressed
  int trigger_released = 0;        // State assigned to when trigger not presed (released)

  std::string state_topic = "";  // Topic for the state of the switch, values = {0,1,2}
  int state_pos_label = 0;       // Positve state  on the RC remote
  int state_neg_label = 2;       // Negative state on the RC remote
  int state_neutral_label = 1;   // Neutral state trigger on the RC remote
};

/// @brief Collision mapper node
class CollisionMapperRos
{
public:
  CollisionMapperRos(ros::NodeHandle &nh, ros::NodeHandle &p_nh);

  // /// We do not allow copies of this class.
  // CollisionMapperRos(const CollisionMapperRos &) = delete;
  // void CollisionMapperRos = (const CollisionMapperRos &) = delete;

  /// @brief Callback to receive the latest trigger msg
  auto rc_trigger_cb(const std_msgs::UInt8 msg) -> void;

  /// @brief Callback to receive the latest trigger msg
  auto rc_toggle_cb(const std_msgs::UInt8 msg) -> void;

private:
  /// @brief Main loop
  auto main_cb(const ros::TimerEvent &) -> void;

  /// @brief computes the collision map
  auto compute_collision_map() -> void;

  /// @brief Loads the ros params
  auto load_ros_params() -> void;

  /// @brief Callback to generates collision states
  auto collision_state_cb(const ros::TimerEvent &) -> void;

  /// @brief Publishes the collision states as pose
  /// @param collision_states
  /// @return void
  auto visualise_collision_trajectory(const std::vector<CollisionState> &collision_states) -> void;

  /// @brief Visualizes the collision clouds
  auto publish_colision_cloud(const std::vector<double> &map_roi) -> void;

  /// @brief  Callback such for auto CM computation
  auto auto_cm_cb(const ros::TimerEvent &) -> void { compute_collision_map_ = true; };

  /// @brief Service to manually trigger the collision_map callback
  auto request_collision_map(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res) -> bool
  {
    compute_collision_map_ = true;
    return true;
  };

  /// @brief  P
  /// @param collision_states
  /// @return
  auto publish_poses(const std::vector<CollisionState> &collision_states) -> void;

  auto reset_map() -> void;

  /// Caluclates the roi
  auto calculate_roi(double padding) -> std::vector<double>;

  ros::NodeHandle nh_, p_nh_;
  std::shared_ptr<ohm::OccupancyMap> map_;
  std::shared_ptr<nve_lfe::ProbTeLablerROS> labler_;

  ros::Subscriber rc_trigger_subs_, rc_toggle_sub_;
  ros::Publisher pub_collision_states_, pub_collision_cloud_, pub_collision_poses_;
  ros::Timer main_cb_timer_, collision_state_cb_, auto_cm_timer_;
  ros::ServiceServer toggle_collision_map_srv_;

  /// @brief  Avoid copy as unque ptr
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_list_;

  std::vector<CollisionState> collision_states_;
  /// @brief Trigger state enables recoding/logging, rc_collision state defines what state the robot is in
  double trigger_state_{ 0 }, rc_collision_state{ 0 };

  CollisionMapParams params_;

  /// Debug and visualisation params
  bool visualize_collision_states_{ false };
  bool visualize_collision_map_{ false };
  bool visualize_coverage_map{ false };
  bool publish_poses_{ false };
  bool compute_collision_map_{ false };

  /// @brief Params for node similarity calculation
  ros::Time last_cs_update_{ 0.0 }, last_collision_update_{ 0.0 };
  Eigen::Vector3d last_pos_{};
};

CollisionMapperRos::CollisionMapperRos(ros::NodeHandle &nh, ros::NodeHandle &p_nh)
  : nh_(nh)
  , p_nh_(p_nh)
{
  load_ros_params();
  /// Callback for
  rc_trigger_subs_ = nh_.subscribe("rc_trigger", 1, &CollisionMapperRos::rc_trigger_cb, this);
  rc_toggle_sub_ = nh_.subscribe("rc_toggle", 1, &CollisionMapperRos::rc_toggle_cb, this);


  /// Setup timers
  main_cb_timer_ =
    nh_.createTimer(ros::Duration(1.0 / params_.main_cb_rate), &CollisionMapperRos::main_cb, this, false);
  collision_state_cb_ =
    nh_.createTimer(ros::Duration(1.0 / params_.cs_rate), &CollisionMapperRos::collision_state_cb, this, false);
  toggle_collision_map_srv_ =
    nh_.advertiseService("compute_collision_map", &CollisionMapperRos::request_collision_map, this);

  if (params_.auto_cm_interval > 0)
  {
    auto_cm_timer_ =
      nh_.createTimer(ros::Duration(params_.auto_cm_interval), &CollisionMapperRos::auto_cm_cb, this, false);
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
  tf_list_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);


  /// Optional publishers
  pub_collision_states_ = nh.advertise<visualization_msgs::MarkerArray>("collision_states", 10);
  pub_collision_cloud_ = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("collision_cloud", 10);
  pub_collision_poses_ = nh.advertise<nav_msgs::Path>("poses", 10);


  /// Initialize labler and map
  map_ = std::make_shared<ohm::OccupancyMap>(params_.voxel_size);
  reset_map();
  labler_ = std::make_shared<nve_lfe::ProbTeLablerROS>(nh_, p_nh_, map_);
}


auto CollisionMapperRos::load_ros_params() -> void
{
  p_nh_.param("world_frame", params_.world_frame, std::string{ "map" });
  p_nh_.param("odom_frame", params_.odom_frame, std::string{ "odom" });
  p_nh_.param("robot_frame", params_.robot_frame, std::string{ "base_link" });
  p_nh_.param<double>("voxel_size", params_.voxel_size, 0.1);
  p_nh_.param<double>("main_rate", params_.main_cb_rate, 0.5);
  p_nh_.param<double>("auto_cm_interval", params_.auto_cm_interval, -1.0);
  p_nh_.param<double>("cs_rate", params_.cs_rate, -1.0);
  p_nh_.param<double>("dt_pos_cs", params_.dt_pos_cs,
                      1.0);  // How many seconds does there need to be between two positive examples
  p_nh_.param<double>("min_pose_dist", params_.min_pose_dist, 0.3);
  p_nh_.param<double>("dt_collision_reset", params_.dt_collision_reset,
                      1.0);  // How many seconds does there need to be between two positive examples

  p_nh_.param<bool>("visualize_collision_states", visualize_collision_states_, false);
  p_nh_.param<bool>("visualize_collision_map", visualize_collision_map_, false);
}

/// TODO:  Add time stamps so we can avoid dealing with old msgs
auto CollisionMapperRos::rc_trigger_cb(std_msgs::UInt8 msg) -> void
{
  trigger_state_ = msg.data;
}

auto CollisionMapperRos::rc_toggle_cb(std_msgs::UInt8 msg) -> void
{
  rc_collision_state = msg.data;
}

auto CollisionMapperRos::collision_state_cb(const ros::TimerEvent &) -> void
{
  auto tag = "collision_state_cb";
  ROS_DEBUG_THROTTLE_NAMED(30.0, tag, "Collision state loop");

  /// Do not update if neutral label, or a collsion occured n seconds ago, @p `dt_collision_reset`
  if ((rc_collision_state == params_.state_neutral_label) ||
      (ros::Time::now() - last_collision_update_) < ros::Duration(params_.dt_collision_reset))
  {
    ROS_DEBUG_THROTTLE_NAMED(10.0, tag, "No measurements received");
    return;
  }

  /// Get latest transform
  geometry_msgs::TransformStamped transformStamped{};
  try
  {
    transformStamped = tf_buffer_->lookupTransform(params_.odom_frame, params_.robot_frame, ros::Time(0.0));
  }
  catch (tf2::TransformException &ex)
  {
    ROS_WARN_THROTTLE_NAMED(30.0, tag, "Could not transform  %s", ex.what());
    return;
  }

  /// Always record collisions:


  Eigen::Vector3d new_pos =
    Eigen::Vector3d(transformStamped.transform.translation.x, transformStamped.transform.translation.y,
                    transformStamped.transform.translation.z);


  /// Enforcing spatial and temporal minimum distance of mearruementsl.Skip if the poses are to close or the recording
  /// between to collisiont state meassurements is to small.
  bool shoul_skip = ((new_pos - last_pos_).squaredNorm() < params_.min_pose_dist * params_.min_pose_dist) ||
                    ((ros::Time::now() - last_cs_update_) < ros::Duration(params_.dt_pos_cs));

  // Only skip if the meassurment is not a collision
  if (shoul_skip and (rc_collision_state != params_.state_neg_label))
  {
    return;
  }
  /// Update the last_update and position and collision state meassurement
  last_cs_update_ = ros::Time::now();
  last_pos_ = new_pos;

  CollisionState new_collision_state;

  /// Only collisions or NTE are explitly mapped
  /// (if trigger_pressed and state_collision) -> collision
  if (trigger_state_ == params_.trigger_pressed and rc_collision_state == params_.state_neg_label)
  {
    last_collision_update_ = ros::Time::now();
    new_collision_state.label = nve_lfe::TeLabel::Collision;
  }
  else
  {
    new_collision_state.label = nve_lfe::TeLabel::Free;
  }

  /// Add the collision states to the processed
  new_collision_state.pose = tf2::transformToEigen(transformStamped);
  new_collision_state.stamp = transformStamped.header.stamp;
  collision_states_.emplace_back(new_collision_state);
}


auto CollisionMapperRos::main_cb(const ros::TimerEvent &) -> void
{
  std::string tag = "main_cb";
  ROS_DEBUG_THROTTLE_NAMED(30, tag, "Computing collision_map");

  if (compute_collision_map_)
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    map_ = std::make_shared<ohm::OccupancyMap>(params_.voxel_size);
    reset_map();
    auto map_l = labler_->map_.get();
    auto map_here = map_.get();

    labler_->map_ = map_;
    labler_->updateMap(map_);
    compute_collision_map();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dtime_ms = 1.e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    ROS_DEBUG_STREAM_THROTTLE_NAMED(30, tag, "Time taken in ms: " << dtime_ms);

    if (visualize_collision_map_ and pub_collision_cloud_.getNumSubscribers() > 0)
    {
      auto map_roi = calculate_roi(1.0);
      publish_colision_cloud(map_roi);
    }
    compute_collision_map_ = false;
  }


  if (visualize_collision_states_ and pub_collision_states_.getNumSubscribers() > 0)
  {
    visualise_collision_trajectory(collision_states_);
  }


  if (pub_collision_poses_.getNumSubscribers() > 0)
  {
    publish_poses(collision_states_);
  }
}

auto CollisionMapperRos::calculate_roi(double padding) -> std::vector<double>
{
  std::vector<double> map_roi{ -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 };
  Eigen::Vector3d pose_i{};
  for (const auto &state : collision_states_)
  {
    pose_i = state.pose.translation();
    map_roi[0] = pose_i.x() < map_roi[0] ? pose_i.x() - padding : map_roi[0];
    map_roi[1] = pose_i.y() < map_roi[1] ? pose_i.y() - padding : map_roi[1];
    map_roi[2] = pose_i.z() < map_roi[2] ? pose_i.z() - padding : map_roi[2];
    map_roi[3] = pose_i.x() > map_roi[3] ? pose_i.x() + padding : map_roi[3];
    map_roi[4] = pose_i.y() > map_roi[4] ? pose_i.y() + padding : map_roi[4];
    map_roi[5] = pose_i.z() > map_roi[5] ? pose_i.z() + padding : map_roi[5];
  }

  return map_roi;
}

auto CollisionMapperRos::compute_collision_map() -> void
{
  for (const auto &c_state : collision_states_)
  {
    labler_->labelPoseWithTargetLabel(c_state.pose, c_state.label);
  }
}

auto CollisionMapperRos::publish_poses(const std::vector<CollisionState> &collision_states) -> void
{
  nav_msgs::Path poses;
  poses.poses.reserve(collision_states_.size());
  poses.header.stamp = ros::Time::now();
  poses.header.frame_id = params_.odom_frame;

  Eigen::Vector3d pose_i;
  Eigen::Quaterniond quat_i;
  for (const auto &state : collision_states)
  {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.frame_id = params_.odom_frame;
    pose_msg.header.stamp = state.stamp;

    // Create a marker message
    pose_i = state.pose.translation();
    pose_msg.pose.position.x = pose_i.x();
    pose_msg.pose.position.y = pose_i.y();
    pose_msg.pose.position.z = pose_i.z();
    quat_i = Eigen::Quaterniond(state.pose.rotation());
    quat_i.normalize();
    pose_msg.pose.orientation.x = quat_i.x();
    pose_msg.pose.orientation.y = quat_i.y();
    pose_msg.pose.orientation.z = quat_i.z();
    pose_msg.pose.orientation.w = quat_i.w();
    poses.poses.push_back(pose_msg);
  }
  pub_collision_poses_.publish(poses);
}

auto CollisionMapperRos::visualise_collision_trajectory(const std::vector<CollisionState> &collision_states) -> void
{
  visualization_msgs::MarkerArray markers_msgs;
  markers_msgs.markers.reserve(collision_states.size());

  /// Loop over all states and assign the correct colour
  int i = 0;
  Eigen::Vector3d pose_i{};
  Eigen::Quaterniond quat_i{};
  for (const auto &state : collision_states)
  {
    // Create a marker message
    visualization_msgs::Marker marker;
    marker.header.frame_id = params_.odom_frame;      // Set the frame ID
    marker.ns = "collision_poses";                    // Set the namespace
    marker.id = i;                                    // Set the marker ID
    marker.type = visualization_msgs::Marker::ARROW;  // Set the marker type to an arrow
    marker.action = visualization_msgs::Marker::ADD;  // Set the marker action to add

    // Set the pose of the marker
    pose_i = state.pose.translation();
    marker.pose.position.x = pose_i.x();
    marker.pose.position.y = pose_i.y();
    marker.pose.position.z = pose_i.z();
    quat_i = Eigen::Quaterniond(state.pose.rotation());
    marker.pose.orientation.x = quat_i.x();
    marker.pose.orientation.y = quat_i.y();
    marker.pose.orientation.z = quat_i.z();
    marker.pose.orientation.w = quat_i.w();

    // Set the scale of the marker
    marker.scale.x = 0.5;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;

    // Set the color of the marker
    if (state.label == nve_lfe::TeLabel::Collision)
    {
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
    }
    else if (state.label == nve_lfe::TeLabel::Free)
    {
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;
      marker.color.a = 1.0;
    }
    else
    {
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
    }

    // Set the lifetime of the marker (0 means forever)col
    // marker.lifetime = ros::Duration();
    markers_msgs.markers.push_back(marker);
    i++;
  }

  pub_collision_states_.publish(markers_msgs);
}

auto CollisionMapperRos::publish_colision_cloud(const std::vector<double> &map_roi) -> void
{
  /// Update the ROI defined by our trajectory

  auto min_key = map_->voxelKey(glm::dvec3(map_roi[0], map_roi[1], map_roi[2]));
  auto max_key = map_->voxelKey(glm::dvec3(map_roi[3], map_roi[4], map_roi[5]));
  auto key_range = ohm::KeyRange(min_key, max_key, map_->regionVoxelDimensions());

  /// Colourisation for RGBA
  ohm::SemanticLabel label{};
  pcl::PointCloud<pcl::PointXYZI> cloud;

  /// Approxiamtion of how many cells once can expect to be filled, normaly ~ 20% is reasnable
  cloud.points.reserve(
    (std::abs(map_roi[3] - map_roi[0]) + std::abs(map_roi[3] - map_roi[0]) + std::abs(map_roi[3] - map_roi[0])) /
    params_.voxel_size * 0.2);


  for (auto &key : key_range)
  {
    if (not labler_->getSemanticData(key, label))
    {
      continue;
    }
    /// If occupied or has n measurements
    pcl::PointXYZI pcl_point;
    auto pos = map_->voxelCentreGlobal(key);

    pcl_point.x = pos.x;
    pcl_point.y = pos.y;
    pcl_point.z = pos.z;
    pcl_point.intensity = nve_lfe::probFromLogsOdds(label.prob_label);
    /// Colour transform from [0-1] to [0 - 255] in RGB

    // const tinycolormap::Color colour =
    //   tinycolormap::GetColor(nve_lfe::probFromLogsOdds(label.prob_label),
    //   tinycolormap::ColormapType::Viridis);

    // pcl_point.r = colour.r() * 255;
    // pcl_point.g = colour.g() * 255;
    // pcl_point.b = colour.b() * 255;
    // pcl_point.label = label.label;
    cloud.points.emplace_back(pcl_point);
  }

  /// Setup publisher
  if (cloud.points.size() > 0)
  {
    cloud.header.frame_id = params_.odom_frame;
    pcl_conversions::toPCL(ros::Time::now(), cloud.header.stamp);
    pub_collision_cloud_.publish(cloud);
  }
}

auto CollisionMapperRos::reset_map() -> void
{
  ohm::MapLayout new_layout = map_->layout();
  if (new_layout.semanticLayer() < 0)
  {
    ohm::addSemanticLayer(new_layout);
  }
  map_->updateLayout(new_layout);
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "Collision Mapper Node");

  ros::NodeHandle nh("~");
  ros::NodeHandle nh_private("~");

  CollisionMapperRos collision_mapper_ros(nh, nh_private);

  ros::spin();

  return -1;
}
