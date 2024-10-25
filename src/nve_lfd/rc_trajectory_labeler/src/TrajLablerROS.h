#ifndef TAJ_LABLER_ROS_H
#define TAJ_LABLER_ROS_H

#include <nve_tools/TrajectoryLoader.h>

#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include  <ros/ros.h>


namespace nve_lfe
{
using namespace std::chrono_literals;

/// Parameters to be expected to be loaded for this
struct Parameters_t
{
  bool debug;
  bool visualize;
  bool verbose;
  bool remove_non_labled;
  std::vector<std::string> rc_bag_files = { "" };
  std::string traj_file = "";

  std::string trigger_topic = "";  // Topic for state of the trigger button, values can range from {0,2}
  int trigger_pressed = 2;         // State assigned to when trigger pressed
  int trigger_released = 0;        // State assigned to when trigger not presed (released)

  std::string state_topic = "";  // Topic for the state of the switch, values = {0,1,2}
  int state_pos_label = 0;       // Positve state  on the RC remote
  int state_neg_label = 2;       // Negative state on the RC remote
  int state_neutral_label = 1;   // Neutral state trigger on the RC remote

  std::string output_file;  // Location where the file will be saved
  int out_pos_label = 1;
  int out_neg_label = 0;
  int out_neutral_label = -1;
};

// Easy data structure to move stuff around...
/// Left over of script and to lazy to change at this points
struct TrajData
{
  std::vector<double> state_times{};
  std::vector<double> trigger_times{};
  std::vector<int> state_labels;
  std::vector<int> trigger_labels;
};

/// @brief Converts from the rc state labels to a desired output state label
/// @param[In] State
int stateLabelToOutputLabel(int state_label, const Parameters_t &params);


/// TrajectoryLabler node
class TrajectoryLablerNode
{
public:
  TrajectoryLablerNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private);

  auto spinROS()->bool { return !pipeline_; };

private:
  /// @brief Loads all parameteters defined in the Parameters_t, defualt values 
  auto loadRosParams()->void;

  /// @brief Loads the bag and processess it
  auto processBag()->void;

  /// Associates a trajectory label with a pose in traj
  void assignStateLablesToPose(Parameters_t &params, TrajData &data, nve_tools::TrajectoryIO &traj);

  // Timer callback for publishing visualization
  auto timerCb(const ros::TimerEvent &) -> void;

  /// Generating Trajectory Message
  void generateTrajectoryMessage(nve_tools::TrajectoryIO &traj);

  /// Filter the traj based on the positve and negative out labels
  void filterSemanticTrajectory(nve_tools::TrajectoryIO &traj, Parameters_t &params);

  /// PARAMS
  ros::NodeHandle nh_, nh_private_;
  ros::Timer timer_{};
  ros::Publisher publisher_;

  Parameters_t params_;
  TrajData traj_data_;
  visualization_msgs::MarkerArray traj_array_;
  bool pipeline_;
};

}  // namespace nve_lfe
#endif