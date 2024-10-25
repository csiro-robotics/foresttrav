#include "ros/ros.h"
#include "OfflineColourFuserRos.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "ohm_colour_visualiser_node");
  ros::NodeHandle nh{};
  ros::NodeHandle private_nh{ "~" };

  /// Load ohm map
  ros::Publisher cloud_pub = private_nh.advertise<sensor_msgs::PointCloud2>("colour_cloud", 1);

  // Load ohm map
  ohm::OccupancyMap map(0.1, ohm::MapFlag::kDefault);
  std::string ohm_map_file{};
  private_nh.param<std::string>("ohm_file", ohm_map_file, "");
  if (ohm::load(ohm_map_file.c_str(), map, nullptr, nullptr))
  {
    ROS_FATAL_STREAM( "Could not load ohm file. Check if file exists: \n" << ohm_map_file);
  }

  /// Get map bounds and range
  std::vector<double> bounds {-20,-20,-3,20, 20, 3};
  auto min_key = map.voxelKey(glm::dvec3(bounds[0], bounds[1], bounds[2]));
  auto max_key = map.voxelKey(glm::dvec3(bounds[3], bounds[4], bounds[5]));
  auto key_range = ohm::KeyRange(min_key, max_key, map.regionVoxelDimensions());

  /// Colourisation for RGBA
  pcl::PointCloud<pcl::PointXYZRGBL> rgb_cloud;
  rgb_cloud.points.reserve(2e5);
  ohm::Voxel<ohm::VoxelMean> mean_layer(&map, map.layout().meanLayer());
  ohm::Voxel<ohm::AppearanceVoxel> rgb_layer(&map, map.layout().appearanceLayer());
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
  cloud_msg.header.frame_id = "map";
  cloud_msg.header.stamp = ros::Time::now();

  ros::Rate r(1);
  while(nh.ok())
  {

    cloud_pub.publish(cloud_msg); 
    ros::spinOnce();
    r.sleep();
  }
  
  return 0;
}