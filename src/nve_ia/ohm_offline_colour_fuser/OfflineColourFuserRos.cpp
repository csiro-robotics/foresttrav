#include "OfflineColourFuserRos.h"

namespace nve_ros
{

OhmOfflineColourFuserNode::OhmOfflineColourFuserNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
  : nh_(nh)
  , nh_private_(nh_private)
  , image_transport_(nh)
{
  load_ros_params();

  // Add colour layer
  auto color_layer_id = map_.layout().appearanceLayer();
  if (0 > color_layer_id)
  {
    ohm::MapLayout new_layout = map_.layout();
    ohm::addAppearanceLayer(new_layout);
    map_.updateLayout(new_layout);
  }

  /// 
  vis_check_.set_camera_matrix(camera_matrix_);
  fuser_.setColouFusionMode(mode_, beta_);
  if (not pipeline_)
  {
    pub_img_ = image_transport_.advertise("image_topic", 1);
    local_pcl_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>("colour_cloud", 1);
    full_pcl_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>("full_colour_cloud", 1);

    main_timer_cb_ = nh_.createTimer(ros::Duration(1 / rate_), &OhmOfflineColourFuserNode::main_cb, this, false);
    
    if(debug_){
      debug_timer_cb_ = nh_.createTimer(ros::Duration(5.0), &OhmOfflineColourFuserNode::published_fused_colour_cloud, this, false);
      debug_visible_points_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>("visible_points", 1);
    }
  }
}


void OhmOfflineColourFuserNode::main_cb(const ros::TimerEvent &)
{
  auto b = process_next_frame();
}

bool OhmOfflineColourFuserNode::process_next_frame()
{
  /// Update reader
  end_of_video_ = not cap_.read(img_);
  img_idx_++;
  /// Process the next frame
  while (0 != (img_idx_ % step_size_))
  {
    end_of_video_ = not cap_.read(img_);

    img_idx_++;

    if (end_of_video_)
    {
      return true;
    }
  }

  /// To modulate step size we need to skip the frames we dont want but still read them....
  double timestamp = image_ts_[img_idx_];
  if (img_idx_ >= image_ts_.size())
  {
    ROS_WARN_STREAM("Reached end of video");
    return true;
  }

  /// Get Transforms
  update_transforms(timestamp);

  //   /// Get visible points and colours
  std::vector<glm::dvec3> points_in_ts;
  ray_buffer_.get_points_around(timestamp, points_in_ts);

  std::vector<Eigen::Vector3d> points_e3d;
  Eigen::Vector3d pi_c;
 
  for (auto &p : points_in_ts)
  {
    pi_c = T_c_m_ * Eigen::Vector3d(p.x, p.y, p.z);
    if (vis_check_.point_inside_frustum(pi_c))
    {
      points_e3d.push_back(Eigen::Vector3d(p.x, p.y, p.z));
    }
  }

  if (points_e3d.empty())
  {
    return false;
  }

  /// Colour fuser
  auto sensor_org = T_m_c_ * Eigen::Vector3d(0, 0, 0);

  fuser_.integrate_rgb_colour(map_, img_, sensor_org, points_e3d, T_c_m_);
  auto points = fuser_.get_endpoints();
  auto colours = fuser_.get_endpoints_colour();

  /// DEBUG Visualizations
  if (not pipeline_)
  {
    publish_tf();
    publish_img();
    publish_colorized_points(points, colours);

    if(debug_)
    { 
      publish_visible_points(points_e3d);
    }

  }

  return false;
}

void OhmOfflineColourFuserNode::save_colourized_map()
{
  std::string ohm_map_file{}, out_file{};
  nh_private_.param<std::string>("ohm_file", ohm_map_file, "");
  nh_private_.param<std::string>("path_out_file", out_file, "");

  std::string out_file_final = "";
  if (out_file.empty())
  {
    out_file = ohm_map_file.substr(0, ohm_map_file.size() - 4) + "_rgb.ohm";
  }
  else
  {
    out_file = out_file;
  }
  ohm::save(out_file, map_);
  ROS_INFO_STREAM("Colorized ohm file saved to: " << out_file);
}


void OhmOfflineColourFuserNode::update_transforms(const double timestamp)
{
  T_m_r_ = robotPose(timestamp);
  T_m_c_ = T_m_r_ * T_r_c_;
  T_c_m_ = T_m_c_.inverse();
}

Eigen::Affine3d OhmOfflineColourFuserNode::robotPose(const double timestamp)
{  /// Get Camera Pose
  glm::dvec4 quat_glm;
  glm::dvec3 pos_glm;
  auto idx = traj_.nearestPose(timestamp, pos_glm, quat_glm);

  Eigen::Translation3d trans_k = Eigen::Translation3d(pos_glm.x, pos_glm.y, pos_glm.z);
  Eigen::Quaterniond quat_k; 
  quat_k.w() = quat_glm.w; 
  quat_k.x() = quat_glm.x;
  quat_k.y() = quat_glm.y; 
  quat_k.z() = quat_glm.z;
  quat_k.normalize();
  Eigen::Isometry3d T_m_r = Eigen::Isometry3d(trans_k * quat_k);

  return T_m_r;
}

void OhmOfflineColourFuserNode::load_ros_params()
{
  std::string node_name_ = "[ColourFuserNode]: ";
  nh_private_.param<bool>("debug", debug_, false);
  nh_private_.param<bool>("pipeline", pipeline_, true);
  nh_private_.param<double>("rate", rate_, 10.0);
  // Params
  auto map_res{ 0.1 };
  nh_private_.param<double>("map_res", map_res, 0.1);
  nh_private_.getParam("map_bounds", map_bounds_);
  if (map_bounds_.empty())
  {
    map_bounds_ = { -1.0, -1.0, -3.0, 1.0, 1.0, 3 };
  }
    nh_private_.getParam("img_mask", img_mask_);
  if (img_mask_.empty())
  {
    img_mask_ = {0.1, 0.9, 0.0, 1.0 };
  }


  std::string ohm_map_file{};
  nh_private_.param<std::string>("ohm_file", ohm_map_file, "");
  if (ohm::load(ohm_map_file.c_str(), map_, nullptr, nullptr))
  {
    ROS_FATAL_STREAM(node_name_ << "Could not load ohm file. Check if file exists: \n" << ohm_map_file);
  }

  std::string traj_file{};
  nh_private_.param<std::string>("traj_file", traj_file, "");
  if (!traj_.load(traj_file))
  {
    ROS_FATAL_STREAM(node_name_ << "Could not load traj file. Check if file exists: \n" << traj_file);
  }

  /// Point Cloud Buffer
  double inner_ts{}, outer_ts{};
  std::string pcl_file_path{};
  nh_private_.param<double>("buffer_inner_ts", inner_ts, 0.05);
  nh_private_.param<double>("buffer_outer_ts", outer_ts, 10.0);
  nh_private_.param<std::string>("pcl_file", pcl_file_path, "");
  if (!ray_buffer_.open_file(pcl_file_path, inner_ts, outer_ts))
  {
    ROS_FATAL_STREAM(node_name_ << "Could not open pcl file. Check if it exists: \n " << pcl_file_path);
  }

  /// Camera Model
  nh_private_.getParam("camera_matrix", camera_matrix_);
  if (camera_matrix_.empty())
  {
    camera_matrix_ = { 0.0, 0.0, 0.0, 0.0 };
    ROS_FATAL_STREAM(node_name_ << "Camera Matrix not provided");
  }

  nh_private_.getParam("camera_dist", camera_dist_coeff_);
  if (camera_dist_coeff_.empty())
  {
    camera_dist_coeff_ = { 0.0, 0.0, 0.0, 0.0 };
    ROS_FATAL_STREAM(node_name_ << "Camera Distrotion Matrix not provided");
  }

  /// Transfrom from robot camera to base_link frame [x,y,z, qw, qx, qy. qz] [m]
  std::vector<double> vec_t_r_c{};
  nh_private_.getParam("T_r_c", vec_t_r_c);
  if (vec_t_r_c.empty())
  {
    ROS_FATAL_STREAM(node_name_ << "Transform from camera to base link not provided");
  }

  auto trans_k = Eigen::Translation3d(vec_t_r_c[0], vec_t_r_c[1], vec_t_r_c[2]);
  Eigen::Quaterniond quat_k = Eigen::Quaterniond(vec_t_r_c[3], vec_t_r_c[4], vec_t_r_c[5], vec_t_r_c[6]);
  quat_k.normalize();
  T_r_c_ = Eigen::Isometry3d(trans_k * quat_k);

  // Video Files IO
  std::string video_file_path{}, video_file_timestamp_path{};
  nh_private_.param<std::string>("video_file", video_file_path, "");
  cap_.open(video_file_path);
  if (!cap_.isOpened())
  {
    ROS_FATAL_STREAM(node_name_ << "Could not load video file. Check if file exists: \n" << video_file_path);
  }

  nh_private_.param<std::string>("video_timestamp_file", video_file_timestamp_path, "");
  if (video_file_timestamp_path.empty())
  {
    ROS_FATAL_STREAM(node_name_ << "Could not load video timestamp file. Check if file exists: \n"
                                << video_file_timestamp_path);
  }

  nve_tools::ImageTSLoader image_ts_loader;
  image_ts_loader.load(video_file_timestamp_path);
  image_ts_ = image_ts_loader.times_dc();

  nh_private_.param<int>("step_size", step_size_, 1);
  fuser_.set_camera_matrix(camera_matrix_, camera_dist_coeff_);
  fuser_.set_image_mask(img_mask_);

  // /// TODO: Add start time here?
  img_idx_ = 0;
  cap_.set(cv::CAP_PROP_POS_FRAMES, img_idx_+1);  // Set the inital image

  /// Full path to out file
  nh_private_.param<std::string>("path_out_file", "");
  int mode;
  nh_private_.param<int>("colour_mode", mode);
  mode_ = static_cast<nve_core::ColourFusionMode>(mode);

  nh_private_.param<double>("colour_param", beta_, 1.0);

  nh_private_.param<std::string>("world_frame", world_frame_, "map");
  nh_private_.param<std::string>("robot_frame", robot_frame_, "base_link");
  nh_private_.param<std::string>("camera_frame", camera_frame_, "cam0");

  if (true)
  {
    ROS_INFO_STREAM(node_name_ << " [pipeline]: " << pipeline_);
    ROS_INFO_STREAM(node_name_ << " [rate]: " << rate_);
    ROS_INFO_STREAM(node_name_ << " [map_res]: " << map_res);
    ROS_INFO_STREAM(node_name_ << " [map_bounds]: " << map_bounds_[0] << ", " << map_bounds_[1] << ", "
                               << map_bounds_[2] << ", " << map_bounds_[3] << ", " << map_bounds_[4] << ", "
                               << map_bounds_[5]);
    ROS_INFO_STREAM(" ");
    ROS_INFO_STREAM(node_name_ << " [RAYCLOUD BUFFER] " << pipeline_);
    ROS_INFO_STREAM(node_name_ << " [coloud_mode] " << mode << " " << mode_);
    ROS_INFO_STREAM(node_name_ << " [buffer_outer_ts] " << outer_ts);
    ROS_INFO_STREAM(node_name_ << " [buffer_outer_ts] " << outer_ts);
    ROS_INFO_STREAM(node_name_ << " [CAMERA PARAMS] " << pipeline_);

    ROS_INFO_STREAM(" ");
    ROS_INFO_STREAM(node_name_ << " [FUSER PARAMS] " << mode);
  }
}

void OhmOfflineColourFuserNode::publish_tf()
{
  // RCLCPP_INFO(this->get_logger(),broadcasting transform");
  auto tf_T_m_r = tf2::eigenToTransform(T_m_r_);
  tf_T_m_r.header.stamp = ros::Time::now();
  tf_T_m_r.header.frame_id = world_frame_;
  tf_T_m_r.child_frame_id = robot_frame_;

  auto tf_T_r_c = tf2::eigenToTransform(T_r_c_);
  tf_T_r_c.header.stamp = ros::Time::now();
  tf_T_r_c.header.frame_id = robot_frame_;
  tf_T_r_c.child_frame_id = camera_frame_;

  if (tf_broadcaster_ == nullptr)
  {  // initialize with shared_pointer to 'this' Node
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>();
  }
  tf_broadcaster_->sendTransform(tf_T_m_r);
  tf_broadcaster_->sendTransform(tf_T_r_c);

  return void();
}

void OhmOfflineColourFuserNode::publish_img()
{
  sensor_msgs::ImagePtr msg;
  std_msgs::Header header;
  header.frame_id = camera_frame_;
  header.stamp = ros::Time::now();
  msg = cv_bridge::CvImage(header, "bgr8", img_).toImageMsg();
  pub_img_.publish(msg);
}

void OhmOfflineColourFuserNode::publish_colorized_points(const std::vector<Eigen::Vector3d> &points,
                                                         const std::vector<Eigen::Vector3d> &colours)
{
  if (points.empty()) 
  {
    ROS_WARN_THROTTLE(3, "[CoulourFusionNode]: No valid points found");
    return void();
  }
  if (colours.empty())
  {
    ROS_WARN_THROTTLE(3, "[CoulourFusionNode]: No valid colour found");
    return void();
  }

  pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
  size_t idx = 0;
  pcl::PointXYZRGB point_i;
  for (auto &pi : points)
  {
    point_i.x = pi.x();
    point_i.y = pi.y();
    point_i.z = pi.z();
    point_i.r = colours[idx][0] * 255.0;
    point_i.g = colours[idx][1] * 255.0;
    point_i.b = colours[idx][2] * 255.0;
    pcl_cloud.points.push_back(point_i);
    idx++;
  }

  pcl::toROSMsg(pcl_cloud, cloud_);
  cloud_.header.frame_id = world_frame_;
  cloud_.header.stamp = ros::Time::now();
  local_pcl_pub_.publish(cloud_);
}

void OhmOfflineColourFuserNode::publish_visible_points(const std::vector<Eigen::Vector3d> &points)
{
  if (points.empty())
  {
    return void();
  }
  pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
  size_t idx = 0;
  pcl::PointXYZI point_i;
  for (const auto &pi : points)
  {
    point_i.x = pi.x();
    point_i.y = pi.y();
    point_i.z = pi.z();
    point_i.intensity = 75;
    pcl_cloud.points.push_back(point_i);
    idx++;
  }

  pcl::toROSMsg(pcl_cloud, visible_points_ );
  visible_points_.header.frame_id = world_frame_;
  visible_points_.header.stamp = ros::Time::now();
  debug_visible_points_pub_.publish(visible_points_);
}


void OhmOfflineColourFuserNode::published_fused_colour_cloud(const ros::TimerEvent &)
{
  /// Get map bounds and range
  Eigen::Vector3d pos_min = T_m_r_.translation() + Eigen::Vector3d(map_bounds_[0],map_bounds_[1] ,map_bounds_[2]);
  Eigen::Vector3d pos_max = T_m_r_.translation() + Eigen::Vector3d(map_bounds_[3],map_bounds_[4], map_bounds_[5]);
  auto min_key = map_.voxelKey(glm::dvec3(pos_min[0], pos_min[1], pos_min[2]));
  auto max_key = map_.voxelKey(glm::dvec3(pos_max[0], pos_max[1], pos_max[2]));
  auto key_range = ohm::KeyRange(min_key, max_key, map_.regionVoxelDimensions());


  /// Colourisation for RGBA
  pcl::PointCloud<pcl::PointXYZRGBL> rgb_cloud;
  rgb_cloud.points.reserve(2e5);
  ohm::Voxel<ohm::VoxelMean> mean_layer(&map_, map_.layout().meanLayer());
  ohm::Voxel<ohm::AppearanceVoxel> rgb_layer(&map_, map_.layout().appearanceLayer());
  for (const auto &key : key_range)
  {
    ohm::setVoxelKey(key, mean_layer, rgb_layer);

    if (!mean_layer.isValid() || !rgb_layer.isValid())
    {
      continue;
    }
    const auto pos = ohm::positionSafe(mean_layer);

    const auto colour_voxel = rgb_layer.data();

    if (colour_voxel.count < 2)
    {
      continue;
    }
    pcl::PointXYZRGBL point_i;
    point_i.x = pos.x;
    point_i.y = pos.y;
    point_i.z = pos.z;
    point_i.r = colour_voxel.red[5] * 255.0;
    point_i.g = colour_voxel.green[5] * 255.0;
    point_i.b = colour_voxel.blue[5] * 255.0;
    rgb_cloud.points.push_back(point_i);
  }
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(rgb_cloud, cloud_msg);
  cloud_msg.header.frame_id = world_frame_;
  cloud_msg.header.stamp = ros::Time::now();
  full_pcl_pub_.publish(cloud_msg);
}

}  // namespace nve_ros
