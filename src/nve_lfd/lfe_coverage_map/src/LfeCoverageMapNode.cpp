#include <lfe_coverage_map/LfeCoverageMapNode.h>

#include <costmap_2d/costmap_2d_publisher.h>

namespace nve_lfe
{
LfeCoverageNode::LfeCoverageNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
  : nh_(nh)
  , nh_private_(nh_private)
  , buffer_()
  , tf_listener_(buffer_)
{
  /// @p Load_ROS_PARAMS
  double res{};
  nh_private_.param("map_res", res, 0.1);
  nh_private_.param("map_bounds", map_bounds_, std::vector<double>{ -30.0, -30.0, 30.0, 30.0 });
  nh_private_.param("robot_footprint", rc_bounding_, std::vector<double>{ -1.0, -0.3, 1.0, 0.3 });

  const auto map_dx = std::abs(map_bounds_[2] - map_bounds_[0]);
  const auto map_dy = std::abs(map_bounds_[3] - map_bounds_[1]);
  const auto nx = map_dx / res;
  const auto ny = map_dy / res;
  const auto x_org = map_dx * 0.5 + map_bounds_[0];
  const auto y_org = map_dy * 0.5 + map_bounds_[1];

  /// Get frames
  nh_private_.param("robot_frame", frames_.robot_frame, std::string{ "slam_base_link" });
  nh_private_.param("base_link", frames_.base_link, std::string{ "slam_base_link" });
  nh_private_.param("world_frame", frames_.world_frame, std::string{ "map" });

  costmap_ = std::make_shared<costmap_2d::Costmap2D>(map_dx / res, map_dy / res, res, map_bounds_[0], map_bounds_[1]);

  /// Initialize costmap2d
  double rate = 0.1;
  nh_private_.param("rate", rate, 0.1);
  main_timer_ = nh_private_.createTimer(ros::Duration(1 / rate), &LfeCoverageNode::main_cb, this, false);

  /// Publisher
  std::string topic{ "coverage_map" };
  costmap_pub_node_ =
    std::make_shared<costmap_2d::Costmap2DPublisher>(&nh_private, costmap_.get(), frames_.world_frame, topic, true);

  // Timer to publish the map
  double map_pub_rate{};
  nh_private_.param("map_pub_rate", map_pub_rate, 1.0);
  map_pub_timer_ = nh_private_.createTimer(
    ros::Duration(1 / map_pub_rate), [&](const ros::TimerEvent & /*unused*/) { costmap_pub_node_->publishCostmap(); },
    false, true);
}


auto LfeCoverageNode::main_cb(const ros::TimerEvent &) -> void
{
  // Get current Pose
  geometry_msgs::TransformStamped transformStamped;
  try
  {
    transformStamped = buffer_.lookupTransform(frames_.world_frame, frames_.robot_frame, ros::Time(0.0));
  }
  catch (tf2::TransformException &ex)
  {
    ROS_WARN_THROTTLE_NAMED(5.0, ros::this_node::getName(), "Could not transform  %s", ex.what());
    return;
  }
  auto T_MR = tf2::transformToEigen(transformStamped);

  /// Generate Footprint
  std::vector<geometry_msgs::Point> polygon;
  geometry_msgs::Point point_m;
  auto pi_m = T_MR * Eigen::Vector3d(rc_bounding_[0], rc_bounding_[1], 0.0);
  point_m.x = pi_m[0];
  point_m.y = pi_m[1];
  point_m.z = 0;
  polygon.emplace_back(point_m);

  pi_m = T_MR * Eigen::Vector3d(rc_bounding_[2], rc_bounding_[1], 0.0);
  point_m.x = pi_m[0];
  point_m.y = pi_m[1];
  point_m.z = 0;
  polygon.emplace_back(point_m);

  pi_m = T_MR * Eigen::Vector3d(rc_bounding_[2], rc_bounding_[3], 0.0);
  point_m.x = pi_m[0];
  point_m.y = pi_m[1];
  point_m.z = 0;
  polygon.emplace_back(point_m);

  pi_m = T_MR * Eigen::Vector3d(rc_bounding_[0], rc_bounding_[3], 0.0);
  point_m.x = pi_m[0];
  point_m.y = pi_m[1];
  point_m.z = 0;
  polygon.emplace_back(point_m);


  // DEBUG CODE
  for (unsigned int i = 0; i < polygon.size(); ++i)
  {
    costmap_2d::MapLocation loc;
    auto x = costmap_->getSizeInCellsX();
    auto y = costmap_->getSizeInCellsY();
    auto ox = costmap_->getOriginX();
    auto oy = costmap_->getOriginY();
    if (!costmap_->worldToMap(polygon[i].x, polygon[i].y, loc.x, loc.y))
    {
      // ("Polygon lies outside map bounds, so we can't fill it");
      continue;
    }
    auto b = 2;
  }
  /// Update Map
  auto a = costmap_->setConvexPolygonCost(polygon, 120);
  if (not costmap_->setConvexPolygonCost(polygon, 120))
  {
    ROS_WARN("Reached map bounds");
  }
}


}  // namespace nve_lfe