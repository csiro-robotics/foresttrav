#include "ros/ros.h"
#include "OfflineColourFuserRos.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "OfflineColourFuserNode");
  ros::NodeHandle nh{};
  ros::NodeHandle private_nh{ "~" };

  nve_ros::OhmOfflineColourFuserNode feature_map_node(nh, private_nh);


  if(not feature_map_node.pipeline())
  {
    ros::spin();
  }
  else{

    // Ufly way to process pipeline
    ROS_INFO_STREAM("[ColourFuserNode]: Running colourfusion in pipeline mode");
    auto total_frame_numbers = feature_map_node.get_total_number_frames();
    while(not feature_map_node.process_next_frame())
    {
      ROS_INFO_STREAM_THROTTLE(10, "[ColourFuserNode]: Processing freames " << feature_map_node.current_frame() << " / " << total_frame_numbers);
      ros::spinOnce();
    }
    ROS_INFO_STREAM("[ColourFuserNode]: Finished processing and saving colourized map");
    feature_map_node.save_colourized_map();

  }

  return 0;
}
