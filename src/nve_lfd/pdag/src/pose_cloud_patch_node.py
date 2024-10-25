#!/usr/bin/env python

from pathlib import Path
import datetime as dt

import joblib
import numpy as np
import numpy.lib.recfunctions as rf
import ros_numpy as rnp
import pandas as pd

import rospy

# from nve_eval.nve_svm_eval import NveTeData
from sensor_msgs.msg import PointCloud2
import tf2_ros as tf2_ros
import tf2_geometry_msgs
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class PoseCloudPatchNode:
    """ Subscribes to a point cloud and saves patches around the robot to a csv file (or ply?) 
    """
    def __init__(self):

        self.received_new_data_ = False     # Flag to check if new data has been received
        self.data_ = []                     # Original feature data
        self.field_names = []               # Name of point cloud fields
        self.T_target_source = any          # Transfrom if necessary  

        # Incoming point cloud topic
        self.subscription_ = rospy.Subscriber("~cloud_in", PointCloud2, self.pcl_cb)
        
        # Load the ros params
        self.data_root_dir = rospy.get_param("~data_root_dir", "/data/debug/")
        self.data_set_name = rospy.get_param("~data_set_name","undefinded")
        self.target_frame = rospy.get_param("~target_frame", "odom")

        self.setup_io()

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def setup_io(self):
        """ Generates the directories and file paths for the class
        
        File system definition:
        data_root_dir:
            |_<experiment_number>_experiment_<data_set_name>
                |_> (optional) transform_<target_frame_id>_<source_frame_id>.csv
                |_> cloud/
                    |_<time_s>_<time_ns>_feature_cloud.csv/ply  
        """
        # Setup experiment root dir if it does not exist
        self.pose_cloud_dir = Path(self.data_root_dir) / f"adj_ohm_scans_v" 

        # if not self.data_root_dir.exists():
        #     self.data_root_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment and encode data set
        # number = len(list(self.data_root_dir.iterdir()))
        # self.pose_cloud_dir = Path(self.data_root_dir) / f"{number}_exp_{self.data_set_name}"
        if not self.pose_cloud_dir.exists():
            self.pose_cloud_dir.mkdir(parents=True, exist_ok=True)

        #TODO(optional p3): Save all files and params to a log file for the experiment


    def pcl_cb(self, msg):
        """Callback to process the received msg"""

        
        # Check if we need to transform the point cloud and get the transform. If not avaible we a
        # DOes not work well because the point types are not what python expects, i.e. additional fields
        # should_transform =  msg.header.frame_id != self.target_frame:
        
        # if should_transform:
        try:
            self.T_target_source = self.tf_buffer.lookup_transform(self.target_frame, msg.header.frame_id, rospy.Time(0))
            cloud_transformed = do_transform_cloud(msg, self.T_target_source)
            self.data_ = rnp.numpify(cloud_transformed)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr_throttle(5, "Could not find transform")
            return
        
        # self.data_ =  rnp.numpify(msg)

        # Ready data for processing 
        self.field_names = [ x.name for x in msg.fields ]
        self.time = msg.header.stamp
        self.received_new_data_ = True
        file_name = self.pose_cloud_dir / f"{self.time.to_nsec()}_cloud.csv"
        pd.DataFrame(data=self.data_[self.field_names], index=None).to_csv(file_name, header=self.field_names,index=False)


    def main_cb(self, event=None):
        """

        """
        # Should we process?
        if not self.received_new_data_ or len(self.data_) < 1:
            return
        
        # Filter clouds? 

        # Are clouds to close in time? 

        # Fast but extra memory?
        # file_name = self.pose_cloud_dir / f"{self.time.to_sec()}_cloud.csv"
        # pd.DataFrame(data=self.data_[self.field_names], index=None).to_csv(file_name, header=self.field_names,index=False)

        received_new_data = False



def main(args=None):
    rospy.init_node("ndt_te_estimator")

    node = PoseCloudPatchNode()

    rospy.Timer(rospy.Duration(1.0 / 10.0), node.main_cb)

    rospy.spin()


if __name__ == "__main__":
    main()
