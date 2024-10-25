#include "OhmColourFuserRos.h"
#include <ohm/VoxelAppearance.h>
#include <ohm/VoxelMean.h>
#include <sensor_msgs/point_cloud2_iterator.h>


namespace nve_ros
{

OhmColourFuserRos::OhmColourFuserRos(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
  : nh_(nh)
  , nh_private_(nh_private)
{
  // loadRosParams();
  nh_private_.getParam("map_bounds", map_bounds_);
  if (map_bounds_.empty())
  {
    map_bounds_ = { -1.0, -1.0, -3.0, 1.0, 1.0, 3 };
  }

  auto map_res{ 0.1 };
  nh_private_.param<double>("map_res", map_res, 0.1);

  double rate{ 1.0 };
  nh_private_.param<double>("rate", rate, 1.0);
  nh_private_.param<std::string>("world_frame", world_frame_, "map");
  nh_private_.param<std::string>("robot_frame", robot_frame_, "base_link");
  nh_private_.param<std::string>("camera_frame", camera_frame_, "cam0");


  tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
  tf_list_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  /// Set all timers and callbacks

  main_cb_timer_ = nh_.createTimer(ros::Duration(1 / rate), &OhmColourFuserRos::main_cb, this, false);
  rgb_cloud_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>("fusion_rgb_cloud", 1);

  sub_rgb_cloud_ = nh_private_.subscribe("rgb_cloud_in", 5, &OhmColourFuserRos::colour_cloud_cb, this);
  sub_rgb_cloud_compressed_ =
    nh_private_.subscribe("rgb_cloud_compressed_in", 10, &OhmColourFuserRos::colour_cloud_compressed_cb, this);


  // Prepare the ohm data structure
  map_ = std::make_shared<ohm::OccupancyMap>(map_res, ohm::MapFlag::kNone);
  ohm::MapLayout new_layout = map_->layout();
  if (new_layout.appearanceLayer() < 0)
  {
    ohm::addAppearanceLayer(new_layout);
  }
  map_->updateLayout(new_layout);

  fuser_ = std::make_shared<nve_core::ColourFuserEP>(map_);


  ROS_INFO_STREAM_COND(true, "map_bounds: [" << map_bounds_[0] << " " << map_bounds_[1] << " " << map_bounds_[2] << " "
                                             << map_bounds_[3] << " " << map_bounds_[4] << " " << map_bounds_[5]
                                             << " ]");
  ROS_INFO_STREAM_COND(true, " map_res: [" << map_res << " ]");
  ROS_INFO_STREAM_COND(true, " rate: [" << rate << " ]");
}


auto OhmColourFuserRos::main_cb(const ros::TimerEvent &) -> void
{
  if (not map_updated_)
  {
    return;
  }

  pcl::PointCloud<pcl::PointXYZRGB> colour_cloud;
  sensor_msgs::PointCloud2 msg;
  if (not rgbcloud_from_aabb(colour_cloud))
    return;
  pcl::toROSMsg(colour_cloud, msg);
  msg.header.frame_id = world_frame_;
  msg.header.stamp = ros::Time::now();
  rgb_cloud_pub_.publish(msg);
  map_updated_ = false;
}

auto OhmColourFuserRos::colour_cloud_cb(const sensor_msgs::PointCloud2::ConstPtr &msg) -> void
{
  pcl::PointCloud<pcl::PointXYZRGB> colour_cloud;
  pcl::fromROSMsg(*msg, colour_cloud);

  for (auto &point : colour_cloud)
  {
    fuser_->integrate(Eigen::Vector3d(point.x, point.y, point.z), Eigen::Vector3d(point.r, point.g, point.b));
  }
}

auto OhmColourFuserRos::colour_cloud_compressed_cb(const sensor_msgs::PointCloud2::ConstPtr &msg) -> void
{
  //   if(use_multithread_)
  // {
  //   std::thread t(boost::bind(&ObservationBuffer::processIncomingPointCloud,this, _1), msg);
  //   t.detach();
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
  sensor_msgs::PointCloud2ConstIterator<float> iter_rgb(*msg, "rgb");

  // Refrence for how the compressed rgb value should be encoed
  // PCL2 messages on coloured_points
  // uint32_t argb = 0;
  // argb |= (value_alpha << 24); 3
  // argb |= (value_red << 16);   2
  // argb |= (value_green << 8);  1
  // argb |= (value_blue << 0);   0

  // This is not quite clear what should be done here.
  // TLDR: c++20
  uint8_t rgba[4];
  for (size_t i = 0; i < msg->width; i++)
  {
    std::memcpy(rgba, &*iter_rgb, sizeof rgba);
    fuser_->integrate(
      Eigen::Vector3d(*iter_x, *iter_y, *iter_z),
      Eigen::Vector3d(static_cast<double>(rgba[2]), static_cast<double>(rgba[1]), static_cast<double>(rgba[0])));

    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_rgb;
  }
  map_updated_ = true;
}


auto OhmColourFuserRos::get_latest_position(glm::dvec3 &latest_pos) -> bool
{
  geometry_msgs::TransformStamped transformStamped;
  try
  {
    transformStamped = tf_buffer_->lookupTransform(world_frame_, robot_frame_, ros::Time(0));  // TODO: - parameters
  }
  catch (tf2::TransformException &ex)
  {
    ROS_WARN("%s", ex.what());
    return false;
  }
  latest_pos = glm::dvec3(transformStamped.transform.translation.x, transformStamped.transform.translation.y,
                          transformStamped.transform.translation.z);
  return true;
}


auto OhmColourFuserRos::rgbcloud_from_aabb(pcl::PointCloud<pcl::PointXYZRGB> &cloud) -> bool
{
  glm::dvec3 robot_pos;
  if (not get_latest_position(robot_pos))
    return false;

  auto ohm_min_key = map_->voxelKey(glm::dvec3(map_bounds_[0], map_bounds_[1], map_bounds_[2]) + robot_pos);
  auto ohm_max_key = map_->voxelKey(glm::dvec3(map_bounds_[3], map_bounds_[4], map_bounds_[5]) + robot_pos);
  auto keys = ohm::KeyRange(ohm_min_key, ohm_max_key, map_->regionVoxelDimensions());

  Eigen::Vector3d colour{ 0, 0, 0 }, voxel_pos;
  pcl::PointXYZRGB point_rgb;

  cloud.points.reserve(1e5);

  for (auto &key : keys)
  {
    if (not fuser_->getColourAndPosition(key, voxel_pos, colour))
      continue;


    point_rgb.x = voxel_pos[0];
    point_rgb.y = voxel_pos[1];
    point_rgb.z = voxel_pos[2];
    point_rgb.r = colour[0];
    point_rgb.g = colour[1];
    point_rgb.b = colour[2];
    cloud.points.push_back(point_rgb);
  }
  cloud.points.shrink_to_fit();

  return true;
}



}  // namespace nve_ros