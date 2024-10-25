// Copyright (c) 2024
// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
// Queensland University of Technology (QUT)
//
// Author:Fabio Ruetz
#include "Ohm2dCostmapGenerator.h"

#include <ohm/DefaultLayer.h>
#include <ohm/Key.h>
#include <ohm/KeyRange.h>
#include <ohm/MapSerialise.h>
#include <ohm/OccupancyMap.h>
#include <ohm/Voxel.h>
#include <ohm/VoxelOccupancy.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2_eigen/tf2_eigen.h>

#include <glm/vec3.hpp>

#include <geometry_msgs/TransformStamped.h>


namespace nve_ros
{


Ohm2dCostmapGenerator::Ohm2dCostmapGenerator(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
  : nh_(nh)
  , nh_private_(nh_private)
  , tf_buffer_()
  , tf_list_(tf_buffer_)
{
  /// Load all the ros params
  loadParams();

  /// Default map layout
  /// Note: We use ohm here for the global consistency. Could use any other representation (hashing)
  map_ = std::make_shared<ohm::OccupancyMap>(params_.map_res, ohm::MapFlag::kNone);
  auto new_layout = map_->layout();
  if (new_layout.occupancyLayer() < 0)
  {
    ohm::addOccupancy(new_layout);
  }
  map_->updateLayout(new_layout);

  /// Initializing the ground_mapper. This is the "main" data structure that stores the ground
  ground_mapper_ = std::make_shared<nve_ros::GroundMapInterface>(map_);

  te_pcl_cb_ =
    nh_private_.subscribe<sensor_msgs::PointCloud2>("te_cloud_topic", 5, &Ohm2dCostmapGenerator::teCloudCb, this);

  costmap_pub_ = nh_private_.advertise<pcl::PointCloud<pcl::_PointXYZI>>("te_costmap", 1);

  // dense_costmap_pub_ = nh_private_.advertise<rasg_nav_msgs::DensePointCloud2>("dense_te_costmap", 1);

  main_timer_ = nh_.createTimer(ros::Rate(params_.rate), &Ohm2dCostmapGenerator::mainCb, this, false);

  ROS_INFO("Finished constructor");
}

void Ohm2dCostmapGenerator::loadParams()
{
  nh_private_.param<double>("rate", params_.rate, 5.0);
  nh_private_.param<std::string>("world_frame", params_.world_frame, "odom");
  nh_private_.param<std::string>("robot_frame", params_.robot_frame, "base_link");
  nh_private_.param("local_map_bounds", params_.local_map_bounds,
                    std::vector<double>{ -5.0, -5.0, -1.0, 5.0, 5.0, 1.2 });

  nh_private_.param<double>("map_res", params_.map_res, 0.1);
  nh_private_.param<int>("num_vertical_voxels", params_.num_vertical_voxels, 10);

  /// Additional options depending on virtual costing strategy
  nh_private_.param<bool>("use_virtual", params_.use_virtual, false);
  nh_private_.param<bool>("use_patch_fill", params_.use_patch_fill, false);
  nh_private_.param<bool>("use_patch_flood_fill", params_.use_patch_flood_fill, false);
  nh_private_.param<int>("patch_leaf_size", params_.patch_leaf_size, 2);
  nh_private_.param<int>("min_adj_voxel", params_.min_adj_voxel, 2);

  /// Can only use one of the fill algorithems
  if (params_.use_virtual)
  {
    ROS_WARN_COND(params_.use_patch_fill and params_.use_patch_flood_fill,
                  "[CostmapConverter] Using patch and flood fill. Note patch fill will run initially and then only "
                  "flood fill in a second loop.");

    ROS_FATAL_COND(not(params_.use_patch_fill or params_.use_patch_flood_fill),
                   "[CostmapConverter] Requires virtual surface generation but no method was selected. Use "
                   "\"use_patch_fill\" or \"use_patch_flood_fill\"");
  }

  ROS_INFO("Loaded params");
}


void Ohm2dCostmapGenerator::mainCb(ros::TimerEvent const &)
{
  if (not params_.has_new_te_map)
  {
    return;
  }

  auto robot_bounds = getCurrentBounds();

  auto lock = std::unique_lock<std::mutex>(mutex_);

  /// Generated the costmap with
  MeanColumnCosting(params_.num_vertical_voxels);

  /// Fill in the gaps with virtual cost
  if (params_.use_virtual)
  {
    generateVirtualCost(robot_bounds, params_.patch_leaf_size);
  }

  /// Publish the costmap if subscribed
  if (costmap_pub_.getNumSubscribers() > 0)
  {
    auto cloud = pcl::PointCloud<pcl::PointXYZI>();
    generateDebugGroundCloud(*map_, ground_mapper_->getGroundMap(), cloud);
    pcl_conversions::toPCL(ros::Time::now(), cloud.header.stamp);
    cloud.header.frame_id = msg_frame_id_;
    costmap_pub_.publish(cloud);
  }

  /// Reset the node for the next processes
  params_.has_new_te_map = false;
  ground_mapper_->reset();
}


auto Ohm2dCostmapGenerator::teCloudCb(sensor_msgs::PointCloud2::ConstPtr const &msg) -> void
{
  auto const lock = std::unique_lock<std::mutex>(mutex_);
  auto voxel = ohm::Voxel<float>(map_.get(), map_->layout().occupancyLayer());

  msg_frame_id_ = msg->header.frame_id;
  auto iter_x = sensor_msgs::PointCloud2ConstIterator<float>(*msg, "x");
  auto iter_y = sensor_msgs::PointCloud2ConstIterator<float>(*msg, "y");
  auto iter_z = sensor_msgs::PointCloud2ConstIterator<float>(*msg, "z");
  auto iter_prob = sensor_msgs::PointCloud2ConstIterator<float>(*msg, "prob");

  auto bounds = getCurrentBounds();

  for (auto i = size_t(0); i < msg->width; i++)
  {
    /// TODO: Check if the current point is in roi/region of intrest
    if (not isWithinBounds(glm::dvec3(*iter_x, *iter_y, *iter_z), bounds))
    {  /// iteration
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_prob;
      continue;
    }

    auto const key = map_->voxelKey(glm::dvec3(*iter_x, *iter_y, *iter_z));

    ohm::setVoxelKey(key, voxel);
    if (not voxel.isValid())
    {
      ROS_FATAL("[CostmapConverter]: Non-valid OHM layer");
      continue;
    }
    /// Add 1 to the te_prob, so zero remains unknow
    auto value = *iter_prob + 1.0;
    voxel.write(value);

    /// Set the z value of the ground_map
    ground_mapper_->setLowestZ(key, *iter_z);

    /// iteration
    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_prob;
  }
  params_.has_new_te_map = true;
}


auto Ohm2dCostmapGenerator::generateVirtualCost(std::vector<double> const &roi, size_t patch_leaf_size) -> void
{
  if (patch_leaf_size < 1)
  {
    throw std::invalid_argument("Received a invalid patch size: " + std::to_string(patch_leaf_size));
  }

  // auto voxel = ohm::Voxel<float>(map_.get(), map_->layout().occupancyLayer());
  auto const min_pos = map_->voxelCentreGlobal(map_->voxelKey(glm::dvec3(roi[0], roi[1], roi[2])));
  auto const res = map_->resolution();

  auto virtual_ground_keys = std::vector<ohm::Key>{};
  for (double x = min_pos.x; x < roi[3]; x += res)
  {
    for (double y = min_pos.y; y < roi[4]; y += res)
    {
      // The virtual ground are x,y only. Default values for the virtual surfaces set elsewheree
      auto const key = map_->voxelKey(glm::dvec3(x, y, 0));

      // The case were we did not find a low point but it is in the costmapvirtual_ground_keys
      if (ground_mapper_->hasGroundKey(key))
        continue;

      virtual_ground_keys.push_back(key);
    }
  }

  /// TODO: (fab)
  if (params_.use_patch_fill)
  {
    auto virtual_ground_cell = std::vector<Cell>{};
    virtual_ground_cell.reserve(virtual_ground_keys.size());
    auto remaining_virtual_cells = std::vector<ohm::Key>{};

    for (auto const &voxel_key : virtual_ground_keys)
    {
      auto pos = map_->voxelCentreGlobal(voxel_key);
      auto mean_cost = 0.0;
      auto mean_z = 0.0;
      auto num_adj_voxel = int(0);

      // Go to the adjacent keys
      auto const x_min = pos.x - static_cast<double>(patch_leaf_size) * res;
      auto const x_max = pos.x + static_cast<double>(patch_leaf_size) * res;
      for (auto x = x_min; x <= x_max; x += res)
      {
        auto const y_min = pos.y - static_cast<double>(patch_leaf_size) * res;
        auto const y_max = pos.y + static_cast<double>(patch_leaf_size) * res;
        for (auto y = y_min; y <= y_max; y += res)
        {
          /// Don't need to check the voxel itself, since it is not in the hash_map
          auto const key = map_->voxelKey(glm::dvec3(x, y, 0));
          if ((not ground_mapper_->hasGroundKey(key)))
          {
            continue;
          }

          mean_cost += ground_mapper_->getCost(key);
          mean_z += ground_mapper_->getZ(key);
          num_adj_voxel++;
        }
      }

      // / If there eare not enough neighboors to make a good estimate, do nothing
      if (params_.min_adj_voxel > num_adj_voxel)
      {
        /// Only need this if we intend to use the flood fill afterwards
        if (params_.use_patch_flood_fill)
        {
          remaining_virtual_cells.emplace_back(voxel_key);
        }

        continue;
      }
      mean_cost = mean_cost / static_cast<double>(num_adj_voxel);
      mean_z = mean_z / static_cast<double>(num_adj_voxel);

      virtual_ground_cell.emplace_back(Cell(voxel_key, mean_z, mean_cost, false, false));
    }

    /// Add the virtual cells to the map
    for (auto const &cell_i : virtual_ground_cell)
    {
      ground_mapper_->setCell(cell_i.key, cell_i);
    }

    /// Swap for the
    if (params_.use_patch_flood_fill)
    {
      virtual_ground_keys.swap(remaining_virtual_cells);
    }
  }

  if (params_.use_patch_flood_fill)
  {
    for (auto const &voxel_key : virtual_ground_keys)
    {
      auto pos = map_->voxelCentreGlobal(voxel_key);
      auto mean_cost = double(0.0);
      auto mean_z = double(0.0);
      auto num_adj_voxel = size_t(0);

      // Go to the adjacent keys
      auto const x_min = pos.x - static_cast<double>(patch_leaf_size) * res;
      auto const x_max = pos.x + static_cast<double>(patch_leaf_size) * res;
      for (auto x = x_min; x <= x_max; x += res)
      {
        auto const y_min = pos.y - static_cast<double>(patch_leaf_size) * res;
        auto const y_max = pos.y + static_cast<double>(patch_leaf_size) * res;
        for (auto y = y_min; y <= y_max; y += res)
        {
          /// Don't need to check the voxel itself, since it is not in the hash_map
          const ohm::Key key = map_->voxelKey(glm::dvec3(x, y, 0));
          if ((not ground_mapper_->hasGroundKey(key)))
          {
            continue;
          }

          mean_cost += ground_mapper_->getCost(key);
          mean_z += ground_mapper_->getZ(key);
          num_adj_voxel++;
        }
      }

      // / If there eare not enough neighbors to make a good estimate, do nothing
      if (params_.min_adj_voxel > num_adj_voxel)
      {
        continue;
      }
      mean_cost = mean_cost / static_cast<double>(num_adj_voxel);
      mean_z = mean_z / static_cast<double>(num_adj_voxel);
      ground_mapper_->setCell(voxel_key, mean_z, mean_cost, false, false);
    }
  }
}


auto Ohm2dCostmapGenerator::MeanColumnCosting(int voxel_num) -> void
{
  auto const res = map_->resolution();
  auto voxel = ohm::Voxel<float>(map_.get(), map_->layout().occupancyLayer());

  for (auto &key : ground_mapper_->allKeys())
  {
    auto mean_cost = double{ 0.0 };
    auto num_valid_voxels = int(0);
    auto const voxel_centre = map_->voxelCentreGlobal(key);  // Base ground voxel key

    /// Iterate over the collum and find the best
    for (auto i = 0; i < voxel_num; i++)
    {
      auto voxel_pos = voxel_centre + glm::dvec3(0, 0, static_cast<double>(i) * res);
      auto vox_key = map_->voxelKey(voxel_pos);
      ohm::setVoxelKey(vox_key, voxel);
      if (not voxel.isValid())
      {
        continue;
      }
      auto te_cost = float{ 0.0 };
      voxel.read(&te_cost);

      /// Assumption that the if the semantic voxel not initalized, the label needs to be >= 1.
      if (1.0 > te_cost or 2.0 < te_cost)
      {
        continue;
      }
      mean_cost += static_cast<double>(te_cost) - 1.0;
      num_valid_voxels++;
    }

    /// We do not allow unitialised voxels here.
    if (num_valid_voxels < 1)
    {
      std::logic_error("Received a ground cell with no valid values");
      continue;
    }

    /// Set the cell values
    ground_mapper_->setCell(key, ground_mapper_->getZ(key), mean_cost / static_cast<double>(num_valid_voxels), true,
                            true);
  }
}



auto Ohm2dCostmapGenerator::getCurrentBounds() -> std::vector<double>
{
  auto transformStamped = geometry_msgs::TransformStamped();
  try
  {
    transformStamped = tf_buffer_.lookupTransform(msg_frame_id_, params_.robot_frame, ros::Time(0));
  }
  catch (tf2::TransformException &ex)
  {
    ROS_WARN("Could find transform: %s", ex.what());
  }

  /// Find the ground for the local map bounds
  auto robot_bounds = std::vector<double>{
    params_.local_map_bounds[0] + transformStamped.transform.translation.x,
    params_.local_map_bounds[1] + transformStamped.transform.translation.y,
    params_.local_map_bounds[2] + transformStamped.transform.translation.z,
    params_.local_map_bounds[3] + transformStamped.transform.translation.x,
    params_.local_map_bounds[4] + transformStamped.transform.translation.y,
    params_.local_map_bounds[5] + transformStamped.transform.translation.z,
  };
  return robot_bounds;
}


auto generateDebugGroundCloud(ohm::OccupancyMap &map, std::unordered_map<ohm::Key, Cell> const &ground_map,
                              pcl::PointCloud<pcl::PointXYZI> &cloud) -> void
{
  auto pcl_point = pcl::PointXYZI();
  for (auto const &iter : ground_map)
  {
    auto pos = map.voxelCentreGlobal(iter.second.key);
    pcl_point.x = pos.x;
    pcl_point.y = pos.y;
    pcl_point.z = iter.second.z;
    pcl_point.intensity = iter.second.cost;

    if (pcl_point.intensity > 1.0)
    {
      ROS_WARN_STREAM_THROTTLE(10, "[CostmapConverter]: Traversability cost exceeded 1.0 or is below 0.0 in the "
                                   "costmap. This should not be the case");
    }

    cloud.points.emplace_back(pcl_point);
  }
}


template <typename T>
auto isWithinBounds(T const &point, std::vector<double> const &bounds) -> bool
{
  // Check if the bounds has the correct dimensions
  if (bounds.size() != 6)
  {
    std::cerr << "Error: Incorrect dimensions of point or bounds.\n";
    return false;
  }
  return (point[0] >= bounds[0] && point[0] <= bounds[3] && point[1] >= bounds[1] && point[1] <= bounds[4] &&
          point[2] >= bounds[2] && point[2] <= bounds[5]);
}

}  // namespace nve_ros
