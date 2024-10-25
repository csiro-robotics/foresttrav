#include "TrajLablerROS.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/UInt8.h>

namespace nve_lfe
{

TrajectoryLablerNode::TrajectoryLablerNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
  : nh_(nh)
  , nh_private_(nh_private)
  , pipeline_(false)
{
  loadRosParams();

  processBag();

  publisher_ = nh_private_.advertise<visualization_msgs::MarkerArray>("rc_labeled_trajectory", 2);
  timer_ = nh_.createTimer(ros::Duration(1 / 2.0), &TrajectoryLablerNode::timerCb, this, false);
}


auto TrajectoryLablerNode::processBag() -> void
{
  nve_tools::TrajectoryIO traj;
  if (!traj.load(params_.traj_file))
  {
    ROS_FATAL_STREAM("Could not open traj file. Check if it exists: " << params_.traj_file);
    return;
  }

  // Parameters to parse state and trajectory
  std::vector<std::string> topics;
  topics.emplace_back(params_.state_topic);
  topics.emplace_back(params_.trigger_topic);

  // Setup data
  TrajData data{};

  /// ROS1 bag extraction
  for (auto &rc_bag_file : params_.rc_bag_files)
  {
    rosbag::Bag bag;
    bag.open(rc_bag_file, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    BOOST_FOREACH (rosbag::MessageInstance const m, view)
    {
      if (m.getTopic().find(params_.state_topic) != std::string::npos)
      {
        data.state_times.emplace_back(m.getTime().toSec());
        std_msgs::UInt8::ConstPtr msg = m.instantiate<std_msgs::UInt8>();
        data.state_labels.emplace_back(static_cast<int>(msg->data));
      }
      else if (m.getTopic().find(params_.trigger_topic) != std::string::npos)
      {
        data.trigger_times.emplace_back(m.getTime().toSec());
        std_msgs::UInt8::ConstPtr msg = m.instantiate<std_msgs::UInt8>();
        data.trigger_labels.emplace_back(static_cast<int>(static_cast<int>(msg->data)));
      }
    }
    bag.close();
  }

  data.state_times.shrink_to_fit();
  data.state_labels.shrink_to_fit();
  data.trigger_times.shrink_to_fit();
  data.trigger_labels.shrink_to_fit();
  ROS_INFO_STREAM("Found data for " << data.state_labels.size() << " labels, times " << data.state_times.size()
                                    << " for state");
  ROS_INFO_STREAM("Found data for " << data.trigger_times.size() << " labels, times " << data.trigger_times.size()
                                    << " for trigger");

  // Generate labels same size as traj
  std::vector<int> labels;
  labels.reserve(traj.points().size());
  for (size_t i = 0; i < traj.points().size(); i++)
  {
    // Default not know examples
    labels.emplace_back(-1);
  }
  traj.setLabels(labels);

  ROS_INFO_STREAM("Assign State Labels files");
  /// Assign labels
  assignStateLablesToPose(params_, data, traj);

  ROS_INFO_STREAM("Filter files");
  // Filter them if we want to remove the -1 labels
  filterSemanticTrajectory(traj, params_);

  /// Save files
  ROS_INFO_STREAM("Start saving files");

  // Write it 
  if (params_.output_file.empty())
    params_.output_file = params_.traj_file.substr(0, params_.traj_file.size() - 4) + "_labeled.txt";

  if (not traj.write(params_.output_file, params_.remove_non_labled))
  {
    ROS_ERROR_STREAM("File could not be written to : " << params_.output_file);
    ROS_ERROR_STREAM("Number of points: " << traj.points().size());
  }
  else
  {
    ROS_INFO_STREAM("File written to : " << params_.output_file);
  }

  ROS_INFO_STREAM("Generate trajMessgaes");
  if (params_.visualize)
  {
    generateTrajectoryMessage(traj);
  }
}

void TrajectoryLablerNode::loadRosParams()
{
  // Debug flag
  nh_private_.param<bool>("pipeline", pipeline_, false);
  nh_private_.param<bool>("debug", params_.debug, true);
  nh_private_.param<bool>("visualize", params_.visualize, true);
  nh_private_.param<bool>("remove_non_labled", params_.remove_non_labled, true);

  /// IO
  nh_private_.param<std::string>("traj_file", params_.traj_file, "");
  nh_private_.getParam("rc_bag_files", params_.rc_bag_files);
  if (params_.rc_bag_files.empty() || params_.traj_file.empty())
  {
    ROS_FATAL_STREAM("Could not load traj or rc_bag_files");
  }

  /// Parameters for RC trigger stick
  nh_private_.param<std::string>("trigger_topic", params_.trigger_topic, "/squash/logging_trigger");
  nh_private_.param<int>("trigger_pressed", params_.trigger_pressed, 2);
  nh_private_.param<int>("trigger_released", params_.trigger_released, 0);

  /// Parameters for the RC remote state switch
  nh_private_.param<std::string>("state_topic", params_.state_topic, "/squash/logging_state");
  nh_private_.param<int>("state_pos_label", params_.state_pos_label, 0);
  nh_private_.param<int>("state_neg_label", params_.state_neg_label, 2);
  nh_private_.param<int>("state_neutral_label", params_.state_neutral_label, 1);


  /// Files and labels for output
  nh_private_.param<std::string>("output_file", params_.output_file, "");
  nh_private_.param<int>("out_pos_label", params_.out_pos_label, 1);
  nh_private_.param<int>("out_neg_label", params_.out_neg_label, 0);
  nh_private_.param<int>("out_neutral_label", params_.out_neutral_label, -1);


  if (params_.debug)
  {
    std::string node_name = " ";
    ROS_INFO_STREAM("Parameters used: ");
    ROS_INFO_STREAM("Pipeline [ " << pipeline_ << "]");
    ROS_INFO_STREAM("Debug [ " << params_.debug << "]");
    ROS_INFO_STREAM("visualize [ " << params_.visualize << "]");
    ROS_INFO_STREAM("verbose [ " << params_.verbose << "]");
    ROS_INFO_STREAM("remove_non_labled [ " << params_.remove_non_labled << "]");
    ROS_INFO_STREAM("Bag file [ " << params_.rc_bag_files[0] << "]");
    ROS_INFO_STREAM("Traj file [ " << params_.traj_file << "]");

    ROS_INFO_STREAM("Trigger topic[ " << params_.trigger_topic << "], trigger_pressed [" << params_.trigger_pressed
                                      << "], trigger_neg [" << params_.trigger_released << "]");

    ROS_INFO_STREAM("State topic [ " << params_.state_topic << "]");
    ROS_INFO_STREAM("State_pos_label[ " << params_.state_pos_label << "], state_neg_label [" << params_.state_neg_label
                                        << "], state_neutral_label [" << params_.state_neutral_label << "]");

    ROS_INFO_STREAM("output_file [ " << params_.output_file << "]");
    ROS_INFO_STREAM("out_pos_label[ " << params_.out_pos_label << "], out_neg_label [" << params_.out_neg_label
                                      << "], out_neutral_label [" << params_.out_neutral_label << "]");
  }
}

void TrajectoryLablerNode::generateTrajectoryMessage(nve_tools::TrajectoryIO &traj)
{
  auto points = traj.points();
  auto quats = traj.quats();
  auto labels = traj.labels();

  traj_array_.markers.clear();
  traj_array_.markers.reserve(labels.size());

  int j = 0;
  for (int i = 0; i < points.size(); i += 20)
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "trajectory";
    marker.id = j;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = points[i].x;
    marker.pose.position.y = points[i].y;
    marker.pose.position.z = points[i].z;
    marker.pose.orientation.x = quats[i].x;
    marker.pose.orientation.y = quats[i].y;
    marker.pose.orientation.z = quats[i].z;
    marker.pose.orientation.w = quats[i].w;
    marker.scale.x = 0.3;
    marker.scale.y = 0.05;
    marker.scale.z = 0.1;
    marker.color.a = 1.0;  // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    // Change colour based on labels
    /// This should be adatpted to the params
    if (!labels.empty())
    {
      switch (labels[i])
      {
      case 1:  // Traverable
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        break;
      case -1:  // Meh state
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        break;
      case 0:  // Non-traversable
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        break;

      default:  // All pther ones, including -1
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        break;
      }
    }

    traj_array_.markers.emplace_back(marker);
    j++;
  }
}

auto TrajectoryLablerNode::timerCb(const ros::TimerEvent &) -> void
{
  publisher_.publish(traj_array_);
}

void TrajectoryLablerNode::assignStateLablesToPose(Parameters_t &params, TrajData &data, nve_tools::TrajectoryIO &traj)
{
  auto label_i = -1;

  for (size_t i = 0; i < data.state_labels.size(); i++)
  {
    // Filter out by trigger
    if (params.remove_non_labled)
    {
      if ( params.state_neutral_label != data.state_labels[i])
      {
        label_i = stateLabelToOutputLabel(data.state_labels[i], params);
        traj.assignLabelToPose(data.state_times[i], label_i);
      }
    }
    else
    {
      label_i = stateLabelToOutputLabel(data.state_labels[i], params);
      traj.assignLabelToPose(data.state_times[i], label_i);
    }
  }
}

int stateLabelToOutputLabel(int state_label, const Parameters_t &params)
{
  if (state_label == params.state_pos_label)
  {
    return params.out_pos_label;
  }
  else if (state_label == params.state_neg_label)
  {
    return params.out_neg_label;
  }
  else
  {
    return params.out_neutral_label;
  }
}

void TrajectoryLablerNode::filterSemanticTrajectory(nve_tools::TrajectoryIO &traj, Parameters_t &params)
{
  ///
  std::vector<glm::dvec3> new_points;
  std::vector<glm::dvec4> new_quats;
  std::vector<double> new_times;
  std::vector<int> new_labels;

  ROS_INFO_STREAM("Size of points: " << traj.points().size() << ", quats: " << traj.quats().size()
                                     << ", times: " << traj.times().size() << ",labels: " << traj.labels().size());

  for (size_t i = 0; i < traj.points().size(); i++)
  {
    if (traj.labels()[i] == params.out_pos_label || traj.labels()[i] == params.out_neg_label)
    {
      new_points.emplace_back(traj.points()[i]);
      new_quats.emplace_back(traj.quats()[i]);
      new_times.emplace_back(traj.times()[i]);
      new_labels.emplace_back(traj.labels()[i]);
    }
  }
  std::cout << "Filtered trajectory from " << traj.points().size() << " to " << new_points.size()
            << ". Percentage: " << double(new_points.size()) / double(traj.points().size()) << std::endl;
  traj.setPoints(new_points);
  traj.setQuats(new_quats);
  traj.setTimes(new_times);
  traj.setLabels(new_labels);
}


}  // namespace nve_lfe
