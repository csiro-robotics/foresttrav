#ifndef LFD_COVERAGE_NODE_H
#define LFD_COVERAGE_NODE_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <ros/ros.h>
#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/costmap_2d_publisher.h>

#include <vector>

namespace nve_lfe
{
using std::placeholders::_1;
using namespace std::chrono_literals;


struct DefaultFrames
{
  std::string robot_frame{};
  std::string base_link{};
  std::string world_frame{};
};


class LfeCoverageNode
{
public:
  LfeCoverageNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private);

private:
  /// @brief Publishes costmap at a set frequency @p costmap_timer
  void publihs_coverage_map();

  /// @brief Cb to receive the events
  auto main_cb(const ros::TimerEvent &) -> void;

  auto getLatestPosition() -> Eigen::Vector3d;


  ros::NodeHandle nh_, nh_private_;
  ros::Timer main_timer_, map_pub_timer_;
  ros::Publisher coverage_map_pub_;
  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener tf_listener_;

std::shared_ptr<costmap_2d::Costmap2D> costmap_{};
  DefaultFrames frames_{};
  std::vector<double> map_bounds_{}, rc_bounding_{};
  std::shared_ptr<costmap_2d::Costmap2DPublisher> costmap_pub_node_{};
};

}  // namespace nve_lfd
#endif