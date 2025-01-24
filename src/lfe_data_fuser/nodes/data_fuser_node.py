# MIT
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO) 
# Queensland University of Technology (QUT)
#
# Author:Fabio Ruetz

#!/usr/bin/env python
import threading
from pathlib import Path

import h5py
import nav_msgs
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2

import ros_numpy as rnp
from ros_scnn_tm.msg import OnlineDataSet
from datetime import datetime


# TODO: Add TE cloud for comparative evaluation!


class OnlineLfEDataFuserNode:
    """Class to estimate traversability on a per voxel basis"""

    def __init__(self):
        self.load_params()

        self.feature_cloud_keys = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.poses = None
        self.collision_cloud_msg = None
        self.collision_cloud_keys = None

        # Feature cloud params
        self.feature_cloud_msg = None

        self.feature_cloud_buffer = {}
        self.feature_cloud_dt = rospy.Duration(1.0)
        self.last_feature_cloud_time = rospy.Time()
        self.last_collision_map_update = rospy.Time()  # Time for the data set

        # Setup file based on rostime
        self.file_path = None

        # Lock for all the elements
        self.lock = threading.Lock()

        self.subscription_ohm_ = rospy.Subscriber(
            "feature_map", PointCloud2, self.feature_cloud_cb
        )

        self.subscription_cc_ = rospy.Subscriber(
            "collision_map",
            PointCloud2,
            self.collision_cloud_cb,
        )
        self.subscription_poses_ = rospy.Subscriber(
            "collision_poses", nav_msgs.msg.Path, self.pose_cb
        )
        self.pub_debug_cloud = rospy.Publisher(
            "~debug_ohm_cloud", PointCloud2, queue_size=20
        )

        self.pub_new_training_data = rospy.Publisher(
            "~new_training_data", OnlineDataSet, queue_size=1
        )

        # Timers and callbacks need to be last
        self.main_timer = rospy.Timer(
            rospy.Duration(1.0 / self.main_rate), self.main_cb
        )
        self.ohm_collection_timer = rospy.Timer(
            rospy.Duration(1.0 / self.feature_map_rate), self.feature_map_processing_cb
        )

    def load_params(self):
        """Loads all ROS parameters"""
        self.main_rate = rospy.get_param("~main_rate", 1.0)
        self.feature_map_rate = rospy.get_param("~feature_map_rate", 2.0)
        self.voxel_size = rospy.get_param("~voxel_size", 0.1)
        self.data_root_dir = rospy.get_param("~data_root_dir", "")
        self.data_set_name = rospy.get_param("~data_set_name", "incremental_data")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")

        # Feature cloud filtering params
        self.radius = rospy.get_param("~radius", 3.0)
        self.height_low = rospy.get_param("~height_low", -0.5)
        self.height_high = rospy.get_param("~height_high", 0.8)

        # Debug configs:
        self.debug_fm_cloud = rospy.get_param("~debug_fm_cloud", False)

        # Trigger online_training
        self.new_file_per_cycle = rospy.get_param("~new_file_per_cycle", True)
        self.trigger_online_training = rospy.get_param("~trigger_online_training", True)
        self.file_write_mode = rospy.get_param("~'file_write_mode", "w") # Options w, a, ...
        
        # Add unique datetime to avoid polution.
        self.data_root_dir =  Path(self.data_root_dir) / datetime.now().strftime('%Y_%m_%d_%H_%M')

    def generate_new_file(self):
        """Generates the files with the timestamp using ros-time"""
        
        timestamp_str = str(rospy.Time.now().to_sec())
        file_dir = Path(self.data_root_dir)
        file_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = file_dir / f"{self.data_set_name}_{timestamp_str}.hdf5"

    def feature_cloud_cb(self, msg):
        """Callback to process the received point cloud"""

        # Get feature cloud
        self.feature_cloud_msg = msg

        if not self.feature_cloud_keys:
            self.feature_cloud_keys = [field.name for field in msg.fields]

    def collision_cloud_cb(self, msg):
        """Callback to process the received msg"""
        self.collision_cloud_msg = msg

        if not self.collision_cloud_keys:
            self.collision_cloud_keys = [field.name for field in msg.fields]

    def te_cloud_callback(self, msg):
        """Callback to process the received msg"""
        self.te_cloud_msg = msg

    def pose_cb(self, msg):
        """Callback to process the received msg"""
        self.poses = msg

    def feature_map_processing_cb(self, event=None):
        """This callback stores the ohm map at a fixef frequency"""

        # We want the ohm_clouds and the poses
        if not (
            self.feature_cloud_msg
            and self.tf_buffer.can_transform(
                self.odom_frame, self.robot_frame, rospy.Time(0)
            )
        ):
            return

        # TODO: move this into the featue_cloud_cb
        # We want to controll the frequency at which we collect ohm maps
        if (
            self.feature_cloud_msg.header.stamp - self.last_feature_cloud_time
        ) < self.feature_cloud_dt:
            return

        # Get current transform (Avoid catch statement for speed)
        transform = self.tf_buffer.lookup_transform(
            self.odom_frame, self.robot_frame, rospy.Time(0)
        )

        # Crop ohm-cloud around last position received
        ohm_feature_cloud = convert_np_array(
            data_org=rnp.numpify(self.feature_cloud_msg),
            input_features=self.feature_cloud_keys,
        )

        position = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        )
        cropped_cloud = ohm_feature_cloud[
            mask_for_points_within_cylinder(
                ohm_feature_cloud[:, :3],
                position,
                self.radius,
                height_low=self.height_low,
                height_high=self.height_high,
            )
        ]

        # Add the cropped cloud to the buffer
        self.feature_cloud_buffer[self.feature_cloud_msg.header.stamp] = cropped_cloud
        self.last_feature_cloud_time = self.feature_cloud_msg.header.stamp

        rospy.logdebug_throttle(
            10.0, f"OHM map buffer has n msgs {len(self.feature_cloud_buffer.keys())}"
        )

        # Reset until next msgs is received
        self.feature_cloud_msg = None

    def main_cb(self, event=None):
        """Main callback to call all the process to enable the predictions data"""

        # Should we process?
        if not self.collision_cloud_msg:
            return

        rospy.logwarn("Processing collision map and associated data")

        # Try to avoid collisions
        self.lock.acquire()
        feature_buffer_copy = self.feature_cloud_buffer
        self.feature_cloud_buffer = {}
        collision_cloud_msg = self.collision_cloud_msg
        self.collision_cloud_msg = None
        poses = self.poses
        self.poses = []
        self.poses = None
        self.lock.release()

        # Process the data
        if self.new_file_per_cycle or type(self.file_path) is None:
            self.generate_new_file()

        with h5py.File(self.file_path, self.file_write_mode) as file:
            pose_keys = ["x", "y", "z", "qx", "qy", "qz", "qw"]

            # "Intensity is a surreate for label_prob"
            collision_cloud_keys = [
                "label_prob" if item == "intensity" else item
                for item in self.collision_cloud_keys
            ]

            # Deal with the meta data first
            if "meta_data" not in file.keys():
                file.create_dataset(
                    f"meta_data/pose_keys", data=pose_keys, dtype=h5py.string_dtype()
                )
                file.create_dataset(
                    f"meta_data/feature_cloud_keys",
                    data=self.feature_cloud_keys,
                    dtype=h5py.string_dtype(),
                )
                file.create_dataset(
                    f"meta_data/collision_cloud_keys",
                    data=self.collision_cloud_keys,
                    dtype=h5py.string_dtype(),
                )
                file.create_dataset(
                    f"meta_data/parent_frame_id",
                    data=self.odom_frame,
                    dtype=h5py.string_dtype(),
                )
                file.create_dataset(
                    f"meta_data/child_frame_id",
                    data=self.robot_frame,
                    dtype=h5py.string_dtype(),
                )

            # Write ohm clouds
            for key, value in feature_buffer_copy.items():
                file.create_dataset(
                    f"feature_clouds/{rostime_to_string(key)}", data=value
                )

            # Save the collision cloud
            cc_time = collision_cloud_msg.header.stamp
            # Crop ohm-cloud around last position received
            # Note: we need to use the misslabelled collsion cloud keys <intensity> for the numpy conversion
            collision_cloud = convert_np_array(
                data_org=rnp.numpify(collision_cloud_msg),
                input_features=self.collision_cloud_keys,
            )

            file.create_dataset(
                f"collision_clouds/{rostime_to_string(cc_time)}", data=collision_cloud
            )

            # Add all the poses to the
            if poses:
                poses_dict = {
                    pose.header.stamp: np.array(
                        [
                            pose.pose.position.x,
                            pose.pose.position.y,
                            pose.pose.position.z,
                            pose.pose.orientation.x,
                            pose.pose.orientation.y,
                            pose.pose.orientation.z,
                            pose.pose.orientation.w,
                        ]
                    )
                    for pose in poses.poses
                }

                for key, value in poses_dict.items():
                    if f"poses/{rostime_to_string(key)}" in file.keys():
                        continue
                    file.create_dataset(f"poses/{rostime_to_string(key)}", data=value)

            else:
                rospy.logwarn("No poses found")

        # Cleanup what we dont need anymore
        # Esure the flag is empty so we dont run this until we get another collision map
        if self.trigger_online_training:
            msg = OnlineDataSet()
            msg.stamp = self.last_collision_map_update
            msg.file_path = str(self.file_path)
            self.pub_new_training_data.publish(msg)

        self.last_collision_map_update = rospy.Time.now()

    def debug_feature_cloud(self, cloud):
        """ Debug function/callback"""
        data_out_type = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("intensity", "<f4"),
            ]
        )

        cloud_out = np.empty(len(cloud), dtype=data_out_type)
        cloud_out["x"] = cloud[:, 0].view("f4").reshape(cloud.shape[0])
        cloud_out["y"] = cloud[:, 1].view("f4").reshape(cloud.shape[0])
        cloud_out["z"] = cloud[:, 2].view("f4").reshape(cloud.shape[0])
        cloud_out["intensity"] = cloud[:, 3].view("f4").reshape(cloud.shape[0])

        msg = rnp.msgify(PointCloud2, cloud_out)
        msg.header.frame_id = self.feature_cloud_msg.header.frame_id
        msg.header.stamp = rospy.Time.now()
        self.pub_debug_cloud.publish(msg)


def convert_np_array(data_org: np.array, input_features: list, f_dtype=np.float32):
    """Conversts the data into a uniform np.array with float type"""
    dtypes_feature = [(f_i, f_dtype) for f_i in input_features]
    feature_data = (
        data_org[input_features]
        .astype(dtypes_feature)
        .view(f_dtype)
        .reshape(data_org.shape[0], len(dtypes_feature))
    )
    return feature_data


def mask_for_points_within_cylinder(points, center, radius, height_low, height_high):
    """Cylindrical filter for a for a point cloud with arond 'centre'. Returns a boolean np.array (nx1) """
    # Calculate the distances of each point to the center of the cylinder
    distances_xy = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    dist_z = np.abs(points[:, 2] - center[2])

    # Find the points within the radius and height limits of the cylinder
    mask = np.logical_and(
        distances_xy <= radius,
        np.logical_and(height_low <= dist_z, dist_z <= height_high),
    )
    return mask


def rostime_to_string(rostime):
    """Convertes the rostime to string"""
    # Extract timestamp components
    secs = rostime.secs
    nsecs = rostime.nsecs

    # Create a string representation
    time_string = f"{secs}.{nsecs:09d}"
    return time_string


def string_to_rostime(time_string):
    """Converts a string to rostime, expects correct format"""
    # Split the string into seconds and nanoseconds parts
    secs_str, nsecs_str = time_string.split(".")

    # Convert string parts to integers
    secs = int(secs_str)
    nsecs = int(nsecs_str)

    # Create a new rospy.Time object
    rostime = rospy.Time(secs, nsecs)
    return rostime


def main(args=None):
    rospy.init_node("online_collision_mapper")

    node = OnlineLfEDataFuserNode()

    rospy.spin()


if __name__ == "__main__":
    main()
